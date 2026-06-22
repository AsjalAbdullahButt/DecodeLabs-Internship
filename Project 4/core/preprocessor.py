"""
preprocessor.py
===============
Shared image pre-processing utilities for both recognition paths.

This module centralizes every image-manipulation step so that the OCR
pipeline and the object-detection pipeline never re-implement loading,
colour-conversion, blurring, thresholding, deskewing or blob construction
themselves. Two public "entry point" functions are exposed:

    * preprocess_for_ocr(...)        -> binary image ready for Tesseract
    * preprocess_for_detection(...)  -> (blob, original_bgr) for MobileNet-SSD

All images are handled internally in OpenCV's native BGR / numpy format.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_image(image_source):
    """
    Load an image from a file path (str/Path) or a PIL Image object or
    a numpy array. Always returns a numpy array in BGR format (OpenCV standard).
    Raises FileNotFoundError if a path is given but file does not exist.
    Raises ValueError if the image cannot be decoded.
    """
    # Case 1: already a numpy array — assume it is BGR and return unchanged.
    if isinstance(image_source, np.ndarray):
        return image_source

    # Case 2: a PIL Image — convert to numpy (RGB), then flip to BGR for cv2.
    if isinstance(image_source, Image.Image):
        # PIL stores RGB(A); force RGB so the channel count is predictable.
        rgb_array = np.array(image_source.convert("RGB"))
        # cv2 expects BGR, so swap the colour ordering. NEVER hand a PIL
        # image straight to cv2 — that is anti-bug rule #1.
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    # Case 3: a filesystem path (string or pathlib.Path).
    if isinstance(image_source, (str, Path)):
        path = Path(image_source)
        # Fail loudly when the file genuinely does not exist on disk.
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        # cv2.imread returns None on unreadable/corrupt files (anti-bug rule #2).
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Image could not be decoded (corrupt/unsupported): {path}")
        return image

    # Anything else is an unsupported input type.
    raise ValueError(
        f"Unsupported image_source type: {type(image_source)}. "
        "Expected str, pathlib.Path, PIL.Image, or numpy.ndarray."
    )


def to_grayscale(image_bgr):
    """
    Convert BGR image to single-channel grayscale.
    Step 1 of OCR pre-processing: removes color noise that confuses character detection.
    Returns: 2D numpy array (height x width), dtype uint8.
    """
    # If the image is already single-channel, there is nothing to convert.
    if image_bgr.ndim == 2:
        return image_bgr
    # Collapse the three colour channels into one luminance channel.
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(gray_image, kernel_size=(5, 5)):
    """
    Apply Gaussian blur to smooth micro-noise before thresholding.
    kernel_size must be odd integers. Default (5,5) works well for most scanned docs.
    Returns: blurred grayscale image, same shape as input.
    """
    # Validate the kernel: a Gaussian kernel must use positive ODD integers,
    # otherwise OpenCV raises a cryptic error deep in C++.
    if (
        not isinstance(kernel_size, tuple)
        or len(kernel_size) != 2
        or any((not isinstance(k, int)) or k <= 0 or k % 2 == 0 for k in kernel_size)
    ):
        raise ValueError(
            f"kernel_size must be a tuple of two positive odd integers, got {kernel_size}"
        )
    # sigmaX=0 lets OpenCV derive the standard deviation from the kernel size.
    return cv2.GaussianBlur(gray_image, kernel_size, 0)


def apply_otsu_threshold(blurred_image):
    """
    Apply Otsu's binarization: automatically finds the optimal pixel cutoff
    that maximizes inter-class variance between foreground (text) and background.
    Returns: binary image (0 or 255 per pixel), same shape as input.
    """
    # Threshold value passed as 0 because THRESH_OTSU computes it automatically.
    # We use THRESH_BINARY (not _INV) per anti-bug rule #7.
    _computed_threshold, binary = cv2.threshold(
        blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary


def deskew_image(gray_image):
    """
    Detect and correct rotation in a text image using the minimum bounding
    rectangle of all foreground pixels. Critical for tilted scanned documents.
    Returns: rotated image aligned to horizontal baseline.
    If no foreground pixels are found, returns the original image unchanged.
    """
    # Foreground (text) on a binary image after THRESH_BINARY is white (255) on
    # a black background only when inverted; here text may be black on white.
    # We invert so that the "ink" pixels become the non-zero set we measure.
    inverted = cv2.bitwise_not(gray_image)
    coords = cv2.findNonZero(inverted)

    # No ink at all -> nothing to deskew, hand back the original untouched.
    if coords is None:
        return gray_image

    # minAreaRect returns ((cx, cy), (w, h), angle) for the tightest rotated box.
    angle = cv2.minAreaRect(coords)[-1]

    # OpenCV reports the angle in (-90, 0]; normalise it to a small correction.
    if angle < -45:
        angle = 90 + angle  # e.g. -85 deg means the box is nearly vertical

    # Avoid over-rotating images that are already effectively level.
    if abs(angle) < 0.1:
        return gray_image

    # Build a rotation matrix around the image centre and warp.
    (h, w) = gray_image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray_image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,  # extend edge pixels instead of black fill
    )
    return rotated


def preprocess_for_ocr(image_source):
    """
    Full OCR pre-processing pipeline: load -> grayscale -> blur -> threshold -> deskew.
    Returns: preprocessed binary image ready for Tesseract.
    This is the ONLY function OCR pipeline should call for pre-processing.
    """
    bgr = load_image(image_source)          # 1. normalise input to BGR numpy
    gray = to_grayscale(bgr)                # 2. drop colour noise
    blurred = apply_gaussian_blur(gray)     # 3. smooth micro-noise
    binary = apply_otsu_threshold(blurred)  # 4. binarise via Otsu
    deskewed = deskew_image(binary)         # 5. straighten tilted text
    return deskewed


def preprocess_for_detection(image_source, target_size=(300, 300), mean=(127.5, 127.5, 127.5)):
    """
    Prepare a BGR image as a 4D blob for MobileNet-SSD inference.
    target_size: network input dimensions (do not change unless retraining model).
    mean: per-channel mean values for normalization (standard for MobileNet).
    Returns: (blob, original_bgr_image)
      - blob: 4D numpy array of shape (1, 3, 300, 300) for DNN input
      - original_bgr_image: the original image (needed for drawing bounding boxes)
    """
    # Load first so callers can pass paths/PIL/numpy interchangeably.
    original_bgr_image = load_image(image_source)

    # scalefactor = 1/127.5 maps pixel values from [0,255] into [-1,1], which is
    # what MobileNet-SSD expects (anti-bug rule #8). swapRB=True converts the
    # BGR image to RGB because the Caffe model was trained on RGB ordering.
    blob = cv2.dnn.blobFromImage(
        original_bgr_image,
        scalefactor=1 / 127.5,
        size=target_size,
        mean=mean,
        swapRB=True,
        crop=False,
    )
    return blob, original_bgr_image
