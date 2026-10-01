"""Tests for the Testmlbacken ML backend API.

The YOLO model itself is replaced by a stub (no GPU and no weights are
required); everything else - the pipeline, the postprocess functions and the
nvmath math - runs for real.

Run with:

    ```bash
    pytest test_api.py
    ```
"""

import json
from typing import List

import numpy as np
import pytest

import Pipeline
from model import NewModel

LABEL_CONFIG = (
    '<View><Image name="image" value="$image"/>'
    '<RectangleLabels name="label" toName="image">'
    '<Label value="Crack"/>'
    "</RectangleLabels></View>"
)


class _StubTensor:
    """Minimal stand-in for an ultralytics tensor (only what _to_numpy uses)."""

    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self) -> np.ndarray:
        return self._values


class _StubBoxes:
    xywhn = _StubTensor([[0.5, 0.5, 0.2, 0.4]])
    conf = _StubTensor([0.9])
    cls = _StubTensor([0])


class _StubResult:
    boxes = _StubBoxes()
    masks = None
    orig_shape = (100, 200)


class _StubModel:
    """Returns one already-filtered detection, like YOLO26 predict would."""

    names = {0: "Crack"}

    def predict(self, path, **kwargs) -> List:
        assert kwargs["device"] == 0
        assert kwargs["imgsz"] == 960
        assert kwargs["nms"] is False  # native end-to-end (NMS-free) head
        return [_StubResult()]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(Pipeline, "get_model", lambda: _StubModel())
    from _wsgi import init_app

    app = init_app(model_class=NewModel)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_predict(client, tmp_path):
    """POST /predict returns the rectangle produced by the postprocess."""
    image = tmp_path / "image.jpg"
    image.write_bytes(b"x")
    request = {
        "tasks": [{"id": 1, "data": {"image": str(image)}}],
        "label_config": LABEL_CONFIG,
    }

    response = client.post(
        "/predict", data=json.dumps(request), content_type="application/json"
    )

    assert response.status_code == 200
    results = json.loads(response.data)["results"]
    region = results[0]["result"][0]
    assert region["type"] == "rectanglelabels"
    assert region["value"]["rectanglelabels"] == ["Crack"]
    assert abs(region["value"]["x"] - 40.0) < 1e-3
    assert abs(region["value"]["y"] - 30.0) < 1e-3
    assert abs(region["value"]["width"] - 20.0) < 1e-3
    assert abs(region["value"]["height"] - 40.0) < 1e-3
