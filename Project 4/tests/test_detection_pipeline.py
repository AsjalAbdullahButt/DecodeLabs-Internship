"""
test_detection_pipeline.py
==========================
Unit tests for the object-detection path.

Tests that depend on the downloaded model weights or the sample image are
skipped gracefully when those assets are not present, so the suite always runs.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Insert the project root so `core.*` imports resolve from any CWD.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.detection_pipeline import (  # noqa: E402
    load_mobilenet_model,
    detect_objects,
    annotate_detections,
)

PROTOTXT_PATH = PROJECT_ROOT / "models" / "MobileNetSSD_deploy.prototxt"
CAFFEMODEL_PATH = PROJECT_ROOT / "models" / "MobileNetSSD_deploy.caffemodel"
SAMPLE_OBJECTS_PATH = PROJECT_ROOT / "sample_inputs" / "sample_objects.jpg"


def _models_available():
    """True only when both model files and the sample image exist on disk."""
    return (
        PROTOTXT_PATH.exists()
        and CAFFEMODEL_PATH.exists()
        and SAMPLE_OBJECTS_PATH.exists()
    )


def test_load_model_missing_files():
    """Loading non-existent model files must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_mobilenet_model("fake.prototxt", "fake.caffemodel")


def test_detect_objects_returns_list():
    """detect_objects returns a list when the model + sample are available."""
    if not _models_available():
        pytest.skip("Model files or sample image missing — run download_models.py")
    net = load_mobilenet_model(str(PROTOTXT_PATH), str(CAFFEMODEL_PATH))
    detections = detect_objects(net, str(SAMPLE_OBJECTS_PATH))
    assert isinstance(detections, list)


def test_detect_objects_confidence_filter():
    """Every returned detection must meet the default 0.80 confidence floor."""
    if not _models_available():
        pytest.skip("Model files or sample image missing — run download_models.py")
    net = load_mobilenet_model(str(PROTOTXT_PATH), str(CAFFEMODEL_PATH))
    detections = detect_objects(net, str(SAMPLE_OBJECTS_PATH))
    for det in detections:
        assert det["confidence"] >= 0.80


def test_detection_dict_structure():
    """A returned detection must have the documented key structure."""
    if not _models_available():
        pytest.skip("Model files or sample image missing — run download_models.py")
    net = load_mobilenet_model(str(PROTOTXT_PATH), str(CAFFEMODEL_PATH))
    detections = detect_objects(net, str(SAMPLE_OBJECTS_PATH))
    if not detections:
        pytest.skip("No detections returned for the sample image")
    det = detections[0]
    assert set(["label", "confidence", "box"]).issubset(det.keys())
    assert set(["x", "y", "w", "h"]).issubset(det["box"].keys())


def test_annotate_detections_shape():
    """Annotating an empty detection list must not crash and preserves shape."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated = annotate_detections(image, [])
    assert annotated.shape == (480, 640, 3)
