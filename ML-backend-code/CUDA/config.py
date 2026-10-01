"""User configuration for the Testmlbacken ML backend.

Every user-tunable parameter lives here; nothing is passed on the command line
or read from the request.  The fixed inference settings (device = 0,
image size = 960) are deliberately NOT configurable and live in Core_nvmath.py.

Target environment: NVIDIA RTX 50-series GPU with CUDA Toolkit >= 13.0.
"""

# Path/name of the YOLO weights used for inference.
# End-to-end (NMS-free) YOLO26 models only: the pipeline checks for the
# one-to-one detection head at load time and rejects anything else.
# A segmentation model (``*-seg.pt``) is required for the ``mask`` postprocess
# function; detection weights (``*.pt``) only support ``box``.
MODEL_PATH = "yolo26n-seg.pt"

# Postprocess function selection (names registered in Function.Gather):
#   ""                 -> auto: chosen by the annotation kind detected in the
#                         Label Studio label config
#                         (RectangleLabels -> "box_postprocess",
#                          BrushLabels / PolygonLabels -> "mask_postprocess")
#   "box_postprocess"  -> force bounding-box postprocessing
#   "mask_postprocess" -> force mask postprocessing (RLE brush / polygon)
POSTPROCESS_FN = ""

# Extra keyword arguments forwarded to ultralytics ``YOLO.predict()``.
# ``device`` (0) and ``imgsz`` (960) are fixed inside Core_nvmath.py, and
# ``nms=False`` (native end-to-end head) is fixed in Pipeline.py; none of them
# may be added here (the pipeline rejects them).
# ``conf`` is set per image by Pipeline from the Label Studio control-tag
# threshold; the fallback is DEFAULT_SCORE_THRESHOLD below.  Single-class mode:
# no class filtering / label mapping is performed.
PREDICT_PARAMS = {
    "iou": 0.7,
    "max_det": 300,
    "agnostic_nms": False,
    # True returns higher-quality masks at the cost of GPU memory/transfer.
    "retina_masks": False,
    "verbose": False,
}

# Thread pool sizing.
# MAX_THREAD is the pool cap; the pipeline needs 3 stages with one worker each
# (receive / infer / return) plus POSTPROCESS_THREADS workers that share one
# queue and are load-balanced automatically.
MAX_THREAD = 6
POSTPROCESS_THREADS = 2

# Capacity of the queues between pipeline stages (backpressure limit).
QUEUE_MAXSIZE = 64

# Seconds a /predict call waits for its tasks to travel through the pipeline.
PREDICT_TIMEOUT = 120.0

# Default score threshold used when the Label Studio control tag has no
# ``model_score_threshold`` attribute.
DEFAULT_SCORE_THRESHOLD = 0.5
