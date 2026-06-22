"""
streamlit_app.py
================
Main Streamlit UI wiring both recognition paths together.

The user picks a mode in the sidebar (OCR or Object Detection), uploads an
image, and sees the original alongside the annotated result. All model calls
are wrapped in try/except so a failure surfaces as a friendly st.error rather
than crashing the app. The last result is cached in st.session_state so simple
re-runs (e.g. toggling an expander) do not re-process the image.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Make the project root importable so `core.*` modules resolve regardless of
# where Streamlit is launched from.
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.ocr_pipeline import (  # noqa: E402
    check_tesseract_installed,
    extract_text,
    annotate_ocr_result,
    save_ocr_result,
)
from core.detection_pipeline import (  # noqa: E402
    load_mobilenet_model,
    detect_objects,
    annotate_detections,
    save_detection_result,
)

# Model + output paths derived from the project root.
PROTOTXT_PATH = BASE_DIR / "models" / "MobileNetSSD_deploy.prototxt"
CAFFEMODEL_PATH = BASE_DIR / "models" / "MobileNetSSD_deploy.caffemodel"
OUTPUTS_DIR = BASE_DIR / "outputs"

OCR_MODE = "OCR — Text Recognition"
DETECTION_MODE = "Object Detection"

# Human-readable explanation of each PSM mode for the tooltip/help text.
PSM_HELP = (
    "auto: fully automatic page layout  |  "
    "block: one uniform block of text  |  "
    "single_line: a single text line  |  "
    "sparse: scattered text such as invoices/forms"
)


# --------------------------------------------------------------------------- #
# Page configuration & theming
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Project 4 — Image & Text Recognition",
    page_icon="🔍",
    layout="wide",
)

# Custom CSS to apply the requested dark colour scheme.
st.markdown(
    """
    <style>
        .stApp { background-color: #0f1117; color: #f8fafc; }
        section[data-testid="stSidebar"] { background-color: #1c1e26; }
        section[data-testid="stSidebar"] * { color: #f8fafc; }
        h1, h2, h3, h4, h5, h6 { color: #f8fafc; }
        .stButton > button {
            background-color: #6366f1; color: #f8fafc;
            border: none; border-radius: 6px;
        }
        .stButton > button:hover { background-color: #4f46e5; color: #ffffff; }
        .success-text { color: #22c55e; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def bgr_to_rgb(image_bgr):
    """Convert a BGR numpy image to RGB for correct display in Streamlit."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


@st.cache_resource(show_spinner=False)
def get_detection_model():
    """Load the MobileNet-SSD model once and cache it across re-runs."""
    return load_mobilenet_model(PROTOTXT_PATH, CAFFEMODEL_PATH)


def init_session_state():
    """Initialise the keys we use to cache the most recent result."""
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_annotated", None)
    st.session_state.setdefault("last_mode", None)
    st.session_state.setdefault("last_signature", None)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar():
    """Render the sidebar controls and return the user's selections."""
    st.sidebar.title("🧠 Recognition Engine")

    mode = st.sidebar.radio(
        "Select Mode",
        options=[OCR_MODE, DETECTION_MODE],
    )

    psm_mode = None
    if mode == OCR_MODE:
        psm_mode = st.sidebar.selectbox(
            "PSM Mode",
            options=["auto", "block", "single_line", "sparse"],
            help=PSM_HELP,
        )
        st.sidebar.caption(PSM_HELP)

    confidence = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.50,
        max_value=1.00,
        value=0.80,
        step=0.05,
    )

    st.sidebar.divider()
    st.sidebar.caption("Models: pytesseract (OCR) | MobileNet-SSD v3 (Detection)")

    return mode, psm_mode, confidence


# --------------------------------------------------------------------------- #
# OCR flow
# --------------------------------------------------------------------------- #
def run_ocr(pil_image, psm_mode):
    """Run the OCR path, returning (result_dict, annotated_bgr) or (None, None)."""
    installed, message = check_tesseract_installed()
    if not installed:
        st.error(
            "Tesseract OCR engine not found.\n\n"
            "Install it:\n"
            "- Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "- Ubuntu: `sudo apt install tesseract-ocr`\n"
            "- Mac: `brew install tesseract`\n\n"
            f"Details: {message}"
        )
        return None, None

    try:
        # Convert the uploaded PIL image to a BGR numpy array for the pipeline.
        bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
        result = extract_text(bgr, psm_mode=psm_mode)
        annotated = annotate_ocr_result(bgr, result)
        return result, annotated
    except Exception as exc:  # noqa: BLE001 — never let the app crash
        st.error(f"OCR processing failed: {exc}")
        return None, None


def render_ocr_details(result):
    """Show OCR-specific output details inside an expander."""
    with st.expander("Output Details", expanded=True):
        st.metric("Mean Confidence", f"{result['confidence']:.1f}%")
        st.metric("Word Count", result["word_count"])
        st.text_area("Extracted Text", value=result["text"], height=200)


# --------------------------------------------------------------------------- #
# Detection flow
# --------------------------------------------------------------------------- #
def run_detection(pil_image, confidence):
    """Run the detection path, returning (detections, annotated_bgr) or (None, None)."""
    if not PROTOTXT_PATH.exists() or not CAFFEMODEL_PATH.exists():
        st.error(
            "Model files are missing. Run `python models/download_models.py` "
            "to download MobileNet-SSD weights before using detection mode."
        )
        return None, None

    try:
        net = get_detection_model()
        bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
        detections = detect_objects(net, bgr, confidence_threshold=confidence)
        annotated = annotate_detections(bgr, detections)
        return detections, annotated
    except FileNotFoundError as exc:
        st.error(str(exc))
        return None, None
    except Exception as exc:  # noqa: BLE001 — never let the app crash
        st.error(f"Detection processing failed: {exc}")
        return None, None


def render_detection_details(detections):
    """Show detection-specific output details inside an expander."""
    with st.expander("Output Details", expanded=True):
        if not detections:
            st.info("No objects detected above the confidence threshold.")
            return
        # Build a simple table of detections.
        table = [
            {
                "label": d["label"],
                "confidence": f"{d['confidence'] * 100:.1f}%",
                "x": d["box"]["x"],
                "y": d["box"]["y"],
                "w": d["box"]["w"],
                "h": d["box"]["h"],
            }
            for d in detections
        ]
        st.table(table)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    init_session_state()
    mode, psm_mode, confidence = render_sidebar()

    st.header(f"🔍 {mode}")

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
    )

    if uploaded is None:
        st.info("Upload an image to begin.")
        return

    pil_image = Image.open(uploaded)

    # Build a signature so we only re-process when the input actually changes.
    signature = (uploaded.name, mode, psm_mode, confidence, uploaded.size)

    if st.session_state["last_signature"] != signature:
        if mode == OCR_MODE:
            result, annotated = run_ocr(pil_image, psm_mode)
        else:
            result, annotated = run_detection(pil_image, confidence)

        # Cache the freshly computed result.
        st.session_state["last_result"] = result
        st.session_state["last_annotated"] = annotated
        st.session_state["last_mode"] = mode
        st.session_state["last_signature"] = signature

    result = st.session_state["last_result"]
    annotated = st.session_state["last_annotated"]

    # Two columns: original on the left, annotated result on the right.
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Original")
        st.image(pil_image, use_column_width=True)
    with col_right:
        st.subheader("Result")
        if annotated is not None:
            st.image(bgr_to_rgb(annotated), use_column_width=True)
        else:
            st.warning("No result to display.")

    # Output details + save button (only when we have a usable result).
    if result is not None and annotated is not None:
        if mode == OCR_MODE:
            render_ocr_details(result)
        else:
            render_detection_details(result)

        if st.button("Save Result"):
            try:
                if mode == OCR_MODE:
                    saved_path = save_ocr_result(annotated, str(OUTPUTS_DIR))
                else:
                    saved_path = save_detection_result(annotated, str(OUTPUTS_DIR))
                st.markdown(
                    f"<span class='success-text'>Saved to: {saved_path}</span>",
                    unsafe_allow_html=True,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to save result: {exc}")


if __name__ == "__main__":
    main()
