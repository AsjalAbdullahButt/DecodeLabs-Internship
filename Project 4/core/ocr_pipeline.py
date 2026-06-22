"""
ocr_pipeline.py
===============
Path 1 of the recognition project: Optical Character Recognition.

This module wraps Google's Tesseract engine (via pytesseract) and turns a raw
image into a structured result containing the extracted text, a mean
confidence score, a word count and the individual text lines. It also provides
helpers to annotate and persist the OCR output as an image.

Pre-processing is delegated entirely to ``core.preprocessor.preprocess_for_ocr``
so that the load -> grayscale -> blur -> Otsu threshold -> deskew sequence lives
in exactly one place.
"""

import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from core.preprocessor import load_image, preprocess_for_ocr


def _configure_tesseract_cmd():
    """
    Point pytesseract at the Tesseract binary even when it is not on PATH.

    pytesseract shells out to a `tesseract` executable. On Windows the installer
    drops it under Program Files but does not always add it to PATH, so we probe
    the common install locations and set ``tesseract_cmd`` to the first hit.
    If `tesseract` is already resolvable on PATH we leave the default alone.
    """
    # Already on PATH? Nothing to do.
    if shutil.which("tesseract"):
        return

    # Standard Windows install locations (system-wide and per-user).
    candidate_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Tesseract-OCR", "tesseract.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe"),
    ]
    for candidate in candidate_paths:
        if candidate and os.path.exists(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            return


# Resolve the Tesseract binary at import time so every entry point benefits.
_configure_tesseract_cmd()

# Map friendly mode names to Tesseract Page-Segmentation-Mode flags. The UI
# only ever exposes these four keys, and extract_text validates against them.
TESSERACT_CONFIGS = {
    "auto":        "--psm 3",   # Fully automatic layout detection
    "block":       "--psm 6",   # Single uniform block of text (book pages)
    "single_line": "--psm 7",   # One text line (number plates, headers)
    "sparse":      "--psm 11",  # Sparse scattered text (invoices, forms)
}


def check_tesseract_installed():
    """
    Verify Tesseract is installed and accessible on the system PATH.
    Returns: (bool, str) — (True, version_string) or (False, error_message).
    Uses pytesseract.get_tesseract_version(). Catches EnvironmentError and
    FileNotFoundError specifically — do not use bare except.
    """
    try:
        version = pytesseract.get_tesseract_version()
        return True, str(version)
    except (EnvironmentError, FileNotFoundError) as exc:
        # EnvironmentError covers pytesseract's TesseractNotFoundError subclass;
        # FileNotFoundError covers a missing binary on some platforms.
        return False, (
            "Tesseract OCR engine not found. Install it and ensure it is on PATH. "
            f"Original error: {exc}"
        )


def extract_text(image_source, psm_mode="auto", language="eng"):
    """
    Run OCR on a pre-processed image and return structured results.

    Args:
        image_source: file path, PIL Image, or numpy array
        psm_mode: one of "auto", "block", "single_line", "sparse"
        language: Tesseract language code (default "eng")

    Returns dict with keys:
        "text":       str  — raw extracted text (stripped)
        "confidence": float — mean confidence across all detected words (0-100)
        "word_count": int   — number of words found
        "lines":      list  — list of non-empty text lines

    Raises:
        ValueError if psm_mode is not a valid key in TESSERACT_CONFIGS.
        RuntimeError if Tesseract is not installed (call check_tesseract_installed first).
    """
    # Anti-bug rule #10: validate the PSM mode BEFORE building any config string.
    if psm_mode not in TESSERACT_CONFIGS:
        raise ValueError(
            f"Invalid psm_mode '{psm_mode}'. "
            f"Valid options are: {list(TESSERACT_CONFIGS.keys())}"
        )

    # Refuse to run if the engine is missing so callers get a clear error.
    installed, message = check_tesseract_installed()
    if not installed:
        raise RuntimeError(message)

    config = TESSERACT_CONFIGS[psm_mode]

    # Full preprocessing pipeline produces a clean binary image for Tesseract.
    preprocessed = preprocess_for_ocr(image_source)

    # image_to_data returns per-word boxes and confidences as a dict of lists.
    data = pytesseract.image_to_data(
        preprocessed,
        lang=language,
        config=config,
        output_type=pytesseract.Output.DICT,
    )

    # Anti-bug rule #5: rows with conf == -1 are layout metadata, not words.
    confidences = []
    for conf, word in zip(data["conf"], data["text"]):
        conf_value = float(conf)
        # Keep only rows that are real words (conf >= 0) with visible text.
        if conf_value >= 0 and word.strip():
            confidences.append(conf_value)

    mean_confidence = float(np.mean(confidences)) if confidences else 0.0

    # image_to_string gives the cleanest reconstructed text for display.
    raw_text = pytesseract.image_to_string(
        preprocessed, lang=language, config=config
    ).strip()

    # Split into non-empty lines for structured downstream display.
    lines = [line for line in raw_text.splitlines() if line.strip()]

    return {
        "text": raw_text,
        "confidence": round(mean_confidence, 2),
        "word_count": len(confidences),
        "lines": lines,
    }


def annotate_ocr_result(original_image_source, result_dict):
    """
    Draw a green bounding box around the text region on the original image.
    Add a text overlay showing extracted text and confidence.
    Returns: annotated BGR numpy array.
    The original image is NOT modified in place — work on a copy.
    """
    # Load and copy so the caller's original image is never mutated.
    original = load_image(original_image_source)
    annotated = original.copy()

    h, w = annotated.shape[:2]

    # Draw a green rectangle just inside the image borders to frame the text
    # region. (We frame the whole image because Tesseract was run on the full
    # pre-processed page rather than a single detected crop.)
    margin = 5
    cv2.rectangle(
        annotated,
        (margin, margin),
        (w - margin, h - margin),
        (0, 255, 0),  # BGR green
        2,
    )

    # Build a short overlay label: first line of text + confidence.
    first_line = result_dict["lines"][0] if result_dict.get("lines") else ""
    overlay_text = f"{first_line[:40]}  ({result_dict.get('confidence', 0.0):.1f}%)"

    # Draw a filled banner so the overlay text is readable over any background.
    cv2.rectangle(annotated, (0, 0), (w, 30), (0, 128, 0), -1)
    cv2.putText(
        annotated,
        overlay_text,
        (10, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),  # white text
        2,
        cv2.LINE_AA,
    )
    return annotated


def save_ocr_result(annotated_image, output_dir, filename="ocr_result.png"):
    """
    Save annotated image to output_dir/ocr_results/filename.
    Creates directory if it does not exist.
    Returns: full path of saved file as a string.
    """
    # Build the destination under the ocr_results subfolder using pathlib.
    target_dir = Path(output_dir) / "ocr_results"
    target_dir.mkdir(parents=True, exist_ok=True)  # exist_ok avoids races
    output_path = target_dir / filename
    cv2.imwrite(str(output_path), annotated_image)
    return str(output_path)
