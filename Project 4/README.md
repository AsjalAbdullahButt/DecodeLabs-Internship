# Project 4 — Image & Text Recognition
**DecodeLabs Internship | Batch 2026**

## Overview
This project implements two independent recognition pipelines behind a single
Streamlit interface. **Path 1 (OCR)** uses `pytesseract` — a wrapper around
Google's Tesseract engine — to extract printed text from an image after a
deterministic pre-processing chain (grayscale → blur → Otsu threshold →
deskew), reporting a mean per-word confidence score. **Path 2 (Object
Detection)** uses OpenCV's DNN module with a pre-trained MobileNet-SSD model
(transfer learning on the Pascal VOC subset) to draw labelled bounding boxes,
keeping only detections at or above an 80% confidence threshold. Both paths
share one pre-processing module and produce annotated, saveable output images.

## Project Structure
```
project4_recognition/
├── app/
│   └── streamlit_app.py          # Main Streamlit UI — both paths wired here
├── core/
│   ├── __init__.py
│   ├── ocr_pipeline.py           # All OCR logic lives here
│   ├── detection_pipeline.py     # All object detection logic lives here
│   └── preprocessor.py           # Shared image pre-processing utilities
├── models/
│   ├── download_models.py        # Script to auto-download MobileNet-SSD weights
│   ├── MobileNetSSD_deploy.prototxt  # (downloaded by download_models.py)
│   └── MobileNetSSD_deploy.caffemodel # (downloaded by download_models.py)
├── sample_inputs/
│   ├── sample_text.png           # Generated programmatically
│   └── sample_objects.jpg        # Downloaded from a public URL
├── outputs/
│   ├── ocr_results/              # Saved annotated OCR output images
│   └── detection_results/        # Saved annotated detection output images
├── tests/
│   ├── __init__.py
│   ├── test_ocr_pipeline.py      # Unit tests for OCR path
│   └── test_detection_pipeline.py # Unit tests for detection path
├── requirements.txt
├── setup.py
└── README.md
```

## Setup Instructions
1. Clone/download the project.
2. Create a virtual environment: `python -m venv venv`
3. Activate it:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Install the Tesseract OCR engine:
   - **Windows**: download installer from https://github.com/UB-Mannheim/tesseract/wiki
   - **Ubuntu**: `sudo apt install tesseract-ocr`
   - **Mac**: `brew install tesseract`
6. Download model files: `python models/download_models.py`
7. Run the app: `streamlit run app/streamlit_app.py`

## Running Tests
```
pytest tests/ -v
```
Tests that depend on Tesseract, the model weights, or the sample images are
skipped automatically when those assets are not present.

## Recognition Paths

### Path 1: OCR
The OCR pipeline runs a fixed sequence: **load → grayscale → Gaussian blur →
Otsu threshold → deskew → Tesseract**. Grayscale removes colour noise, the blur
suppresses micro-noise, Otsu's method automatically picks the binarization
cutoff that best separates ink from background, and deskewing straightens
tilted scans via the minimum-area bounding rectangle of the foreground pixels.

**PSM (Page Segmentation Mode) options:**
- `auto` (PSM 3) — fully automatic page layout analysis.
- `block` (PSM 6) — a single uniform block of text (e.g. book pages).
- `single_line` (PSM 7) — exactly one line (e.g. number plates, headers).
- `sparse` (PSM 11) — sparse, scattered text (e.g. invoices, forms).

### Path 2: Object Detection
MobileNet-SSD is a single-shot detector: the image is converted into a
`300×300` blob with `blobFromImage`, normalised to `[-1, 1]` using
`scalefactor = 1/127.5` and per-channel mean subtraction, with `swapRB=True` to
match the RGB-trained Caffe model. The forward pass yields normalized corner
coordinates that are scaled into pixel space and clamped to non-negative
values. Only detections with confidence **≥ 0.80** are kept. The model
recognizes the **21-class Pascal VOC label set** (`background` plus 20 object
classes such as person, car, cat, dog, bus, bicycle), not the full 91-class
COCO list.

## Validation Checklist (from project spec)
- [ ] Library Integration: pytesseract and cv2.dnn load without errors
- [ ] Pre-Processing Integrity: grayscale + thresholding demonstrated
- [ ] Accuracy Benchmarking: confidence >= 80% on final output
- [ ] Visual Confirmation: annotated output image generated and saved

## Known Limitations
- Tesseract performs poorly on heavily stylized or handwritten fonts.
- MobileNet-SSD detects 21 VOC classes, not all possible objects.
- Very dark or blurry images reduce OCR accuracy before thresholding helps.
