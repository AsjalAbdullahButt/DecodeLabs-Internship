"""
test_ocr_pipeline.py
====================
Unit tests for the OCR path (preprocessor + ocr_pipeline).

Imports are resolved by inserting the project root at sys.path[0] so the tests
run regardless of the current working directory.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Insert the project root (one level up from tests/) so `core.*` imports work.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.preprocessor import (  # noqa: E402
    load_image,
    to_grayscale,
    preprocess_for_ocr,
)
from core.ocr_pipeline import (  # noqa: E402
    check_tesseract_installed,
    extract_text,
)

SAMPLE_TEXT_PATH = PROJECT_ROOT / "sample_inputs" / "sample_text.png"


def test_check_tesseract_installed():
    """check_tesseract_installed() must return a (bool, str) tuple."""
    result = check_tesseract_installed()
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], bool)
    assert isinstance(result[1], str)


def test_load_image_from_numpy():
    """A numpy array passes through load_image unchanged in shape."""
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    loaded = load_image(arr)
    assert loaded.shape == (100, 100, 3)


def test_load_image_from_path_not_found():
    """A non-existent path must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_image("nonexistent_file.jpg")


def test_to_grayscale_shape():
    """to_grayscale collapses a 3-channel image to 2 dimensions."""
    arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    gray = to_grayscale(arr)
    assert gray.shape == (100, 100)


def test_apply_otsu_threshold_binary():
    """Full OCR preprocessing must yield a strictly binary (0/255) image."""
    if not SAMPLE_TEXT_PATH.exists():
        pytest.skip("sample_text.png not found — run download_models.py first")
    binary = preprocess_for_ocr(str(SAMPLE_TEXT_PATH))
    unique_values = np.unique(binary)
    # Every pixel must be either 0 or 255.
    assert set(unique_values.tolist()).issubset({0, 255})


def test_extract_text_invalid_psm():
    """An invalid PSM mode must raise ValueError before any OCR runs."""
    with pytest.raises(ValueError):
        extract_text(np.zeros((50, 50, 3), dtype=np.uint8), psm_mode="invalid_mode")


def test_extract_text_returns_dict_keys():
    """When Tesseract + sample exist, extract_text returns the documented keys."""
    installed, _ = check_tesseract_installed()
    if not installed:
        pytest.skip("Tesseract not installed")
    if not SAMPLE_TEXT_PATH.exists():
        pytest.skip("sample_text.png not found — run download_models.py first")

    result = extract_text(str(SAMPLE_TEXT_PATH), psm_mode="sparse")
    assert set(["text", "confidence", "word_count", "lines"]).issubset(result.keys())
