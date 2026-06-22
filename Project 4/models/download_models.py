"""
download_models.py
==================
One-shot helper that fetches everything the recognition app needs to run:

    1. MobileNetSSD_deploy.prototxt  (network architecture)
    2. MobileNetSSD_deploy.caffemodel (trained weights)
    3. sample_inputs/sample_objects.jpg (a public test image for detection)
    4. sample_inputs/sample_text.png   (a generated test image for OCR)

Every download is skipped if the target file already exists, so the script is
safe to re-run. All paths are resolved relative to the project root using
pathlib so the script works regardless of the current working directory.
"""

import os
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

# Resolve the project root (one level up from this models/ directory).
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
SAMPLE_DIR = BASE_DIR / "sample_inputs"

# Source URLs as specified by the project brief.
PROTOTXT_URL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
CAFFEMODEL_URL = "https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc"
SAMPLE_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
)

# Destination paths.
PROTOTXT_PATH = MODELS_DIR / "MobileNetSSD_deploy.prototxt"
CAFFEMODEL_PATH = MODELS_DIR / "MobileNetSSD_deploy.caffemodel"
SAMPLE_OBJECTS_PATH = SAMPLE_DIR / "sample_objects.jpg"
SAMPLE_TEXT_PATH = SAMPLE_DIR / "sample_text.png"


def _ensure_dirs():
    """Create the models/ and sample_inputs/ directories if they are missing."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)


def download_file(url, destination, description):
    """
    Stream a URL to disk with a simple text progress indicator.
    Skips the download if the destination file already exists.
    """
    # Skip work that has already been done (idempotent re-runs).
    if os.path.exists(destination):
        print(f"[SKIP] {description} already exists at {destination}")
        return

    print(f"[DOWNLOAD] {description} from {url}")
    try:
        # stream=True avoids loading the whole file into memory at once.
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            downloaded = 0
            chunk_size = 8192

            with open(destination, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Print a coarse percentage progress bar using plain prints.
                    if total > 0:
                        pct = downloaded * 100 // total
                        print(f"    ... {pct}% ({downloaded}/{total} bytes)", end="\r")
                    else:
                        print(f"    ... {downloaded} bytes", end="\r")
        print()  # newline after the progress line
        print(f"[SUCCESS] Saved {description} -> {destination}")
    except requests.RequestException as exc:
        # Clean up a partial file so a later re-run does not "skip" a bad file.
        if os.path.exists(destination):
            os.remove(destination)
        print(f"[ERROR] Failed to download {description}: {exc}")


def generate_sample_text_image():
    """
    Programmatically build sample_inputs/sample_text.png with PIL:
    white 800x300 background and a black invoice-style line of text.
    """
    if os.path.exists(SAMPLE_TEXT_PATH):
        print(f"[SKIP] sample_text.png already exists at {SAMPLE_TEXT_PATH}")
        return

    # White canvas, 800x300, 3-channel RGB.
    image = Image.new("RGB", (800, 300), color="white")
    draw = ImageDraw.Draw(image)
    text = "Invoice #0042  Date: 2024-01-15  Total: $499.00"

    # Try to load a real TrueType font at size 36; fall back to PIL's default.
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 36)
        except (OSError, IOError):
            font = ImageFont.load_default()

    # Draw the text in black, roughly centred vertically.
    draw.text((30, 130), text, fill="black", font=font)
    image.save(SAMPLE_TEXT_PATH)
    print(f"[SUCCESS] Generated sample text image -> {SAMPLE_TEXT_PATH}")


def main():
    """Run the full download + generation sequence."""
    print("=" * 60)
    print("Project 4 — Model & Sample Asset Downloader")
    print("=" * 60)

    _ensure_dirs()

    download_file(PROTOTXT_URL, PROTOTXT_PATH, "MobileNet-SSD prototxt")
    download_file(CAFFEMODEL_URL, CAFFEMODEL_PATH, "MobileNet-SSD caffemodel")
    download_file(SAMPLE_IMAGE_URL, SAMPLE_OBJECTS_PATH, "sample objects image")
    generate_sample_text_image()

    print("=" * 60)
    print("Done. If the caffemodel download failed (Google Drive can block")
    print("automated downloads), download it manually and place it at:")
    print(f"  {CAFFEMODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
