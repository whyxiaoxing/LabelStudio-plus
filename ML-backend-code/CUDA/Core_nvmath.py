"""GPU-side math primitives for the postprocess stage (torch-free).

Inputs and outputs are plain numpy arrays.  ``nvmath-python`` executes the
matrix math on the GPU with ``execution="cuda"`` (cuBLAS) and returns host
arrays; no CPU backend is ever selected.

Detection filtering is *not* done here: confidence, class and max_det are
applied by ``YOLO.predict(conf=..., classes=..., max_det=...)`` inside
ultralytics, so the results reaching postprocessing are already final.  Hence
this module needs no comparison / selection API -- only the coordinate math.

Fixed by design (not user configurable): ``DEVICE = 0`` (first NVIDIA GPU) and
``IMGSZ = 960`` (model input size).  Both are forwarded to ``YOLO.predict`` by
Pipeline.py.
"""

import logging
from typing import List, Tuple

import numpy as np
import nvmath

logger = logging.getLogger(__name__)

# Fixed inference settings -- see module docstring.
DEVICE = 0
IMGSZ = 960
# nvmath math always executes on the GPU (cuBLAS); the CPU backend
# (ExecutionCPU) requires nvmath-python[cpu] and must never be selected.
NVMMATH_EXECUTION = "cuda"

# Homogeneous affine converting normalized center-xywh straight to Label Studio
# percent top-left-xywh: (x, y, w, h) -> (x - w/2, y - h/2, w, h) * 100,
# applied as  out = in @ M.T.  The percent scale is folded in, so the whole
# conversion is a single nvmath matmul.
_XYWHN_TO_LS_PERCENT = np.array(
    [
        [100.0, 0.0, -50.0, 0.0],
        [0.0, 100.0, 0.0, -50.0],
        [0.0, 0.0, 100.0, 0.0],
        [0.0, 0.0, 0.0, 100.0],
    ],
    dtype=np.float32,
)

# Diagonal scale applied to (P, 2) points as an nvmath matmul.
_PERCENT_SCALE_2D = np.array(
    [
        [100.0, 0.0],
        [0.0, 100.0],
    ],
    dtype=np.float32,
)


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a square matrix to ``(N, D)`` row vectors via nvmath on the GPU.

    The transform is ``out = points @ matrix.T`` and execution is forced to
    CUDA.  ``matrix`` must be ``(D, D)``; homogeneous ``(D+1, D+1)`` matrices
    are applied the same way for affine transforms.  Users can reuse this
    helper for custom point math in Function.py.
    """
    points = np.ascontiguousarray(points, dtype=np.float32)
    if points.size == 0:
        return points
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    return nvmath.linalg.matmul(points, matrix.T, execution=NVMMATH_EXECUTION)


def xywhn_to_ls_percent(xywhn: np.ndarray) -> np.ndarray:
    """Convert normalized ``(N, 4)`` center-xywh boxes to Label Studio percent.

    Returns ``(N, 4)`` with x, y, width, height in percent; the top-left shift
    and the percent scale are one nvmath matmul.
    """
    if xywhn.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return transform_points(xywhn, _XYWHN_TO_LS_PERCENT)


def points_to_percent(points: np.ndarray) -> List[List[float]]:
    """Convert one instance's normalized ``(P, 2)`` polygon points to percent.

    The scaling runs through nvmath; the small point matrix is returned as a
    plain list for Label Studio serialization.
    """
    return transform_points(points, _PERCENT_SCALE_2D).tolist()


def mask_to_rle(mask: np.ndarray, orig_shape: Tuple[int, int]) -> str:
    """Encode one host mask to Label Studio RLE (brush) format.

    Run-length encoding is a sequential algorithm that nvmath does not
    provide, so this final step runs on the host with the Label Studio SDK
    helper.  The mask is resized to the original image size and thresholded.
    """
    import cv2
    from label_studio_sdk.converter.brush import mask2rle

    height, width = int(orig_shape[0]), int(orig_shape[1])
    mask = np.asarray(mask)
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype("float32"), (width, height))
    binary = (mask > 0).astype("uint8") * 255
    return mask2rle(binary)
