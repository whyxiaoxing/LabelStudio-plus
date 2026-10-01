"""User-editable postprocess functions.

The functions below receive a ``PostInput`` dataclass (defined in Pipeline.py)
that carries the ultralytics results, and turn them into Label Studio region
dictionaries.  Detection filtering (confidence / max_det) is delegated to
``YOLO.predict``, so nothing is filtered here -- the functions only read the
final instances and hand the coordinate math to Core_nvmath (nvmath on the
GPU, numpy in and out).  Single-class mode: every region carries the control
tag's ``<Label>``.

To customize:
    * add a function, decorate it with ``@Gather.register("<name>")``;
    * point ``config.POSTPROCESS_FN`` at that name to force it, or leave the
      config empty to let the annotation type in the Label Studio label config
      decide (``box`` -> box_postprocess, ``mask`` -> mask_postprocess).
"""

import logging
import uuid
from typing import TYPE_CHECKING, Any, Callable, Dict, List

import numpy as np

from Core_nvmath import mask_to_rle, points_to_percent, xywhn_to_ls_percent

if TYPE_CHECKING:  # only for type hints; avoids a runtime import cycle
    from Pipeline import PostInput

logger = logging.getLogger(__name__)


def _to_numpy(tensor) -> np.ndarray:
    """Read one ultralytics tensor into host memory (simple value access)."""
    return tensor.cpu().numpy()


class Gather:
    """Registry of postprocess functions selected by ``config.POSTPROCESS_FN``."""

    _functions: Dict[str, Callable] = {}
    _KIND_DEFAULT = {"box": "box_postprocess", "mask": "mask_postprocess"}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator registering a postprocess function under ``name``."""

        def decorator(function: Callable) -> Callable:
            cls._functions[name] = function
            return function

        return decorator

    @classmethod
    def get(cls, name: str) -> Callable:
        try:
            return cls._functions[name]
        except KeyError:
            raise KeyError(
                f"Unknown postprocess function '{name}'. Registered: {cls.names()}"
            ) from None

    @classmethod
    def names(cls) -> List[str]:
        return sorted(cls._functions)

    @classmethod
    def default_for_kind(cls, kind: str) -> str:
        try:
            return cls._KIND_DEFAULT[kind]
        except KeyError:
            raise KeyError(
                f"No default postprocess function for annotation kind '{kind}'"
            ) from None

    @classmethod
    def resolve(cls, kind: str) -> str:
        """Return the configured function name, falling back to the kind default."""
        import config

        return config.POSTPROCESS_FN or cls.default_for_kind(kind)


@Gather.register("box_postprocess")
def box_postprocess(item: "PostInput") -> List[Dict[str, Any]]:
    """Turn YOLO detection boxes into Label Studio ``rectanglelabels`` regions.

    No filtering here: ``YOLO.predict(conf=...)`` already returned the final
    detections.  Single-class mode: every region carries the control tag's
    label.  Only the small box arrays are read; the percent conversion runs
    through nvmath.
    """
    spec = item.spec
    boxes = getattr(item.result, "boxes", None)
    if boxes is None:
        return []

    conf = _to_numpy(boxes.conf)
    boxes_xywh = xywhn_to_ls_percent(_to_numpy(boxes.xywhn)).tolist()

    regions = []
    for i, (x, y, width, height) in enumerate(boxes_xywh):
        regions.append(
            {
                "from_name": spec.from_name,
                "to_name": spec.to_name,
                "type": "rectanglelabels",
                "value": {
                    "rectanglelabels": [spec.label],
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                },
                "score": float(conf[i]),
            }
        )
    return regions


@Gather.register("mask_postprocess")
def mask_postprocess(item: "PostInput") -> List[Dict[str, Any]]:
    """Turn YOLO segmentation masks into Label Studio regions.

    ``BrushLabels`` control tags get RLE-encoded masks; ``PolygonLabels`` get
    polygon points.  No filtering here: ``YOLO.predict`` already returned the
    final instances.  Single-class mode: every region carries the control
    tag's label.
    """
    spec = item.spec
    result = item.result
    masks = getattr(result, "masks", None)
    if masks is None:
        return []

    conf = _to_numpy(result.boxes.conf)
    if spec.control_tag == "BrushLabels":
        return _brush_regions(item, masks, conf)
    return _polygon_regions(item, masks, conf)


def _brush_regions(item: "PostInput", masks, conf) -> List[Dict[str, Any]]:
    """RLE (brush) regions from the instance masks."""
    height, width = item.result.orig_shape
    regions = []
    for i in range(len(masks)):
        rle = mask_to_rle(_to_numpy(masks.data[i]), (height, width))
        regions.append(
            {
                "id": str(uuid.uuid4())[:9],
                "from_name": item.spec.from_name,
                "to_name": item.spec.to_name,
                "original_width": width,
                "original_height": height,
                "image_rotation": 0,
                "value": {
                    "format": "rle",
                    "rle": rle,
                    "brushlabels": [item.spec.label],
                },
                "score": float(conf[i]),
                "type": "brushlabels",
            }
        )
    return regions


def _polygon_regions(item: "PostInput", masks, conf) -> List[Dict[str, Any]]:
    """Polygon regions from the instance masks."""
    regions = []
    for i, point_tensor in enumerate(masks.xyn):
        regions.append(
            {
                "from_name": item.spec.from_name,
                "to_name": item.spec.to_name,
                "type": "polygonlabels",
                "value": {
                    "polygonlabels": [item.spec.label],
                    "points": points_to_percent(_to_numpy(point_tensor)),
                    "closed": True,
                },
                "score": float(conf[i]),
            }
        )
    return regions
