"""Inference pipeline for the Testmlbacken ML backend.

Flow (strictly sequential per task, stage to stage through queues)::

    receive -> infer -> postprocess (x2 workers) -> return -> Future

One image per request: Label Studio feeds tasks one at a time, so every
``/predict`` call carries a single task through the pipeline and returns as
soon as that image is done -- results are never accumulated over a batch.

* Queue 1 ``DetectTask``: one Label Studio task entering the pipeline.
* Queue 2 ``InferTask``: media paths resolved, ready for YOLO.
* Queue 3 ``PostTask``: raw ultralytics results, still on the GPU.
* Queue 4 ``ReturnTask``: Label Studio regions ready to send back.

A fixed set of daemon worker threads (1 receive + ``POSTPROCESS_THREADS``
postprocess + 1 infer + 1 return, validated against ``config.MAX_THREAD`` = 6)
runs these stages; the postprocess workers share one queue, so the load
balances between them automatically.  Nothing runs in parallel outside these
stages, and the daemon threads never block interpreter shutdown.

Inference settings are fixed here: ``Core_nvmath.DEVICE`` (0) and
``Core_nvmath.IMGSZ`` (960) are always passed to ``YOLO.predict``; all other
predict parameters come from ``config.PREDICT_PARAMS``.  Detection filtering
(confidence / max_det) is applied inside ``YOLO.predict`` too, not in
postprocessing.  Single-class mode: the control tag's ``<Label>`` is the label
of every produced region; there is no class mapping.
"""

import atexit
import logging
import os
import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import config
import Core_nvmath
from Function import Gather
from label_studio_ml.utils import DATA_UNDEFINED_NAME

logger = logging.getLogger(__name__)

# Control tag -> annotation kind; this is how the annotation type is detected
# automatically from the Label Studio label config.
CONTROL_KIND = {
    "RectangleLabels": "box",
    "BrushLabels": "mask",
    "PolygonLabels": "mask",
}

_QUEUE_TIMEOUT = 0.2
_SENTINEL = object()


@dataclass
class ControlSpec:
    """One Label Studio control tag served by this backend (single class)."""

    from_name: str
    to_name: str
    value: str
    control_tag: str
    kind: str  # "box" | "mask"
    postprocess_fn: str  # name registered in Function.Gather
    score_threshold: float
    label: str  # the single Label Studio label of the control tag


@dataclass
class DetectTask:
    """Queue 1: one Label Studio task entering the pipeline."""

    task: Dict[str, Any]
    specs: List[ControlSpec]
    backend: Any
    future: Future


@dataclass
class InferTask:
    """Queue 2: media paths resolved, ready for the YOLO model."""

    detect: DetectTask
    paths: Dict[str, str]  # control value name -> local file path


@dataclass
class PostTask:
    """Queue 3: raw ultralytics results, still on the GPU."""

    infer: InferTask
    results: Dict[str, Any]  # control value name -> ultralytics Results


@dataclass
class PostInput:
    """Argument passed to the user postprocess functions in Function.py."""

    task: Dict[str, Any]
    spec: ControlSpec
    result: Any
    path: str
    backend: Any


@dataclass
class ReturnTask:
    """Queue 4: Label Studio regions ready to send back."""

    post: PostTask
    regions: List[Dict[str, Any]]


q_detect: "queue.Queue[Any]" = queue.Queue(maxsize=config.QUEUE_MAXSIZE)
q_infer: "queue.Queue[Any]" = queue.Queue(maxsize=config.QUEUE_MAXSIZE)
q_post: "queue.Queue[Any]" = queue.Queue(maxsize=config.QUEUE_MAXSIZE)
q_return: "queue.Queue[Any]" = queue.Queue(maxsize=config.QUEUE_MAXSIZE)

_stop = threading.Event()
_start_lock = threading.Lock()
_workers: List[threading.Thread] = []
_model = None
_model_lock = threading.Lock()


def get_model():
    """Load the YOLO model once per process (lazy, thread-safe).

    End-to-end (NMS-free) models only: YOLO26 weights carry a one-to-one
    detection head, which this pipeline requires.  Anything else is rejected
    here with a clear error -- no fallback to NMS-based inference.
    """
    global _model
    with _model_lock:
        if _model is None:
            from ultralytics import YOLO

            logger.info("Loading YOLO model: %s", config.MODEL_PATH)
            model = YOLO(config.MODEL_PATH)
            if getattr(model.model.model[-1], "one2one_cv2", None) is None:
                raise ValueError(
                    f"End-to-end (NMS-free) model required: '{config.MODEL_PATH}' "
                    "has no one-to-one detection head. Use YOLO26 weights "
                    "(e.g. yolo26n-seg.pt or your trained YOLO26 model); "
                    "YOLOv8/v11 and other NMS-based models are not supported."
                )
            _model = model
        return _model


def detect_controls(backend) -> List[ControlSpec]:
    """Derive the annotation targets from the Label Studio label config.

    The annotation kind is detected automatically from the control tag:
    ``RectangleLabels`` -> box, ``BrushLabels`` / ``PolygonLabels`` -> mask.
    The postprocess function is then either forced by ``config.POSTPROCESS_FN``
    or taken from the kind default (see ``Function.Gather``).

    Single-class mode: each control tag is expected to carry one ``<Label>``;
    its value is used as the label of every produced region.
    """
    specs = []
    for control in backend.label_interface.controls:
        if not control.to_name:
            logger.warning("Control tag %s has no toName, skipping", control.tag)
            continue
        obj = control.objects[0]
        if obj.tag != "Image":
            continue
        kind = CONTROL_KIND.get(control.tag)
        if kind is None:
            continue
        if str(control.attr.get("model_skip", "false")).lower() in ("1", "true", "yes"):
            logger.info("Skipping control tag %s (model_skip=true)", control.name)
            continue
        labels = list(control.labels_attrs or {})
        if not labels:
            logger.warning("Control tag %s has no <Label>, skipping", control.name)
            continue
        if len(labels) > 1:
            logger.warning(
                "Single-class mode: control tag %s has %s labels, using '%s'",
                control.name,
                len(labels),
                labels[0],
            )
        specs.append(
            ControlSpec(
                from_name=control.name,
                to_name=control.to_name[0],
                value=obj.value_name,
                control_tag=control.tag,
                kind=kind,
                postprocess_fn=Gather.resolve(kind),
                score_threshold=float(
                    control.attr.get("model_score_threshold")
                    or config.DEFAULT_SCORE_THRESHOLD
                ),
                label=labels[0],
            )
        )

    if not specs:
        raise ValueError(
            "No box/mask control tags (RectangleLabels, BrushLabels, "
            "PolygonLabels) connected to an Image tag found in the label "
            f"config:\n{backend.label_config}"
        )
    return specs


def _resolve_paths(item: DetectTask) -> Dict[str, str]:
    """Resolve every control value to a local file path (receive stage)."""
    paths: Dict[str, str] = {}
    for spec in item.specs:
        if spec.value in paths:
            continue
        data = item.task.get("data", {})
        raw = data.get(spec.value)
        if raw is None:
            raw = data.get(DATA_UNDEFINED_NAME)
        if raw is None:
            raise ValueError(
                f"Task {item.task.get('id')} has no data value '{spec.value}'"
            )
        if not isinstance(raw, str):
            raise ValueError(
                f"Task data value '{spec.value}' must be a path/URL string, "
                f"got {type(raw).__name__}"
            )
        path = (
            raw
            if os.path.exists(raw)
            else item.backend.get_local_path(raw, task_id=item.task.get("id"))
        )
        logger.debug("Resolved %s -> %s", raw, path)
        paths[spec.value] = path
    return paths


def _get(source: "queue.Queue[Any]") -> Any:
    """Blocking get that still notices shutdown."""
    while not _stop.is_set():
        try:
            return source.get(timeout=_QUEUE_TIMEOUT)
        except queue.Empty:
            continue
    return _SENTINEL


def _fail(task: DetectTask, stage: str, exc: Exception) -> None:
    logger.exception("Pipeline %s stage failed: %s", stage, exc)
    if not task.future.done():
        task.future.set_exception(exc)


def _receive_loop() -> None:
    """Stage 1: resolve media paths for every incoming task."""
    while not _stop.is_set():
        item = _get(q_detect)
        if item is _SENTINEL:
            return
        try:
            q_infer.put(InferTask(detect=item, paths=_resolve_paths(item)))
        except Exception as exc:  # keep the worker alive, fail the request
            _fail(item, "receive", exc)


def _infer_loop() -> None:
    """Stage 2: run YOLO with the fixed device/imgsz plus the conf threshold.

    Detection filtering is delegated to ultralytics: ``conf`` (from the Label
    Studio control tag) is passed to ``YOLO.predict``, so results arriving at
    postprocessing are already final.  Single-class mode: no class filtering.
    """
    while not _stop.is_set():
        item = _get(q_infer)
        if item is _SENTINEL:
            return
        task = item.detect
        try:
            model = get_model()
            results = {}
            for value, path in item.paths.items():
                specs = [spec for spec in task.specs if spec.value == value]
                predict_params = dict(config.PREDICT_PARAMS)
                # The Label Studio control-tag threshold overrides the config
                # default; the lowest threshold wins for tags sharing an image.
                predict_params["conf"] = min(
                    spec.score_threshold for spec in specs
                )
                results[value] = model.predict(
                    path,
                    device=Core_nvmath.DEVICE,
                    imgsz=Core_nvmath.IMGSZ,
                    nms=False,  # fixed: native end-to-end (NMS-free) head
                    **predict_params,
                )
            q_post.put(PostTask(infer=item, results=results))
        except Exception as exc:
            _fail(task, "infer", exc)


def _postprocess_loop() -> None:
    """Stage 3: two workers share this loop and pick tasks off one queue."""
    while not _stop.is_set():
        item = _get(q_post)
        if item is _SENTINEL:
            return
        task = item.infer.detect
        try:
            regions: List[Dict[str, Any]] = []
            for spec in task.specs:
                result = item.results.get(spec.value)
                if result is None:
                    continue
                post_input = PostInput(
                    task=task.task,
                    spec=spec,
                    result=result[0],  # one image per path -> one Results object
                    path=item.infer.paths[spec.value],
                    backend=task.backend,
                )
                regions.extend(Gather.get(spec.postprocess_fn)(post_input))
            q_return.put(ReturnTask(post=item, regions=regions))
        except Exception as exc:
            _fail(task, "postprocess", exc)


def _return_loop() -> None:
    """Stage 4: assemble the Label Studio prediction and complete the future."""
    while not _stop.is_set():
        item = _get(q_return)
        if item is _SENTINEL:
            return
        task = item.post.infer.detect
        try:
            scores = [region["score"] for region in item.regions if "score" in region]
            prediction = {
                "result": item.regions,
                "score": sum(scores) / max(len(scores), 1),
                "model_version": str(task.backend.model_version),
            }
            if not task.future.done():
                task.future.set_result(prediction)
        except Exception as exc:
            _fail(task, "return", exc)


def _validate_config() -> None:
    workers = 3 + int(config.POSTPROCESS_THREADS)
    if workers > int(config.MAX_THREAD):
        raise ValueError(
            f"config.MAX_THREAD={config.MAX_THREAD} cannot host receive+infer+"
            f"return plus POSTPROCESS_THREADS={config.POSTPROCESS_THREADS} workers"
        )
    forbidden = {"device", "imgsz", "source", "stream", "nms"} & set(config.PREDICT_PARAMS)
    if forbidden:
        raise ValueError(
            f"config.PREDICT_PARAMS must not override fixed parameters: {sorted(forbidden)}"
        )


def ensure_started() -> None:
    """Start the worker threads once per process (idempotent)."""
    global _workers
    with _start_lock:
        if _workers:
            return
        _validate_config()
        loops = (
            [_postprocess_loop] * int(config.POSTPROCESS_THREADS)
            + [_receive_loop, _infer_loop, _return_loop]
        )
        _workers = [
            threading.Thread(
                target=loop,
                name=f"mlb-{loop.__name__}-{index}",
                daemon=True,
            )
            for index, loop in enumerate(loops)
        ]
        for worker in _workers:
            worker.start()
        logger.info(
            "Pipeline started: %s workers (max_thread=%s, postprocess threads=%s)",
            len(_workers),
            config.MAX_THREAD,
            config.POSTPROCESS_THREADS,
        )


def stop() -> None:
    """Signal the workers to exit (also registered at interpreter shutdown)."""
    _stop.set()
    for source in (q_detect, q_infer, q_post, q_return):
        try:
            source.put_nowait(_SENTINEL)
        except queue.Full:
            pass


atexit.register(stop)


def run_task(backend, task: Dict[str, Any]) -> Dict[str, Any]:
    """Send one Label Studio task (single image) through the pipeline.

    Returns the prediction dict as soon as the image has travelled the stages;
    the Flask request thread only waits on this one future.  Batching is
    deliberately not supported: Label Studio feeds tasks one at a time.
    """
    ensure_started()
    specs = detect_controls(backend)
    future: Future = Future()
    q_detect.put(
        DetectTask(
            task=task,
            specs=specs,
            backend=backend,
            future=future,
        )
    )
    return future.result(timeout=config.PREDICT_TIMEOUT)
