"""
detection_pipeline.py
=====================
Path 2 of the recognition project: Object Detection.

This module loads a pre-trained MobileNet-SSD Caffe model through OpenCV's DNN
module and runs single-shot detection on an image. Detections below the
configured confidence threshold (default 0.80) are discarded. Helpers are
provided to draw the resulting bounding boxes and to persist the annotated
image.

The model was trained on the 21-class Pascal VOC label set (not the full
91-class COCO list), so ``COCO_LABELS`` below intentionally contains 21 entries.
"""

from pathlib import Path

import cv2
import numpy as np

from core.preprocessor import load_image, preprocess_for_detection

# The 21-class VOC label set MobileNet-SSD was trained on. Index 0 is the
# implicit "background" class. Do NOT pad this list to 91 entries.
COCO_LABELS = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]


def load_mobilenet_model(prototxt_path, caffemodel_path):
    """
    Load MobileNet-SSD model from disk using OpenCV's DNN module.
    Raises FileNotFoundError with a clear message if either file is missing,
    telling the user to run models/download_models.py first.
    Returns: cv2.dnn_Net object.
    """
    prototxt = Path(prototxt_path)
    caffemodel = Path(caffemodel_path)

    # Verify BOTH files are present before asking OpenCV to parse them, so the
    # error message is actionable rather than a cryptic cv2 parse failure.
    if not prototxt.exists() or not caffemodel.exists():
        missing = []
        if not prototxt.exists():
            missing.append(str(prototxt))
        if not caffemodel.exists():
            missing.append(str(caffemodel))
        raise FileNotFoundError(
            "Missing model file(s): "
            + ", ".join(missing)
            + ". Run 'python models/download_models.py' first to download weights."
        )

    # readNetFromCaffe parses the network architecture (.prototxt) and the
    # trained weights (.caffemodel) into a runnable DNN graph.
    net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
    return net


def detect_objects(net, image_source, confidence_threshold=0.80):
    """
    Run MobileNet-SSD inference on an image.

    Args:
        net: loaded cv2.dnn_Net
        image_source: file path, PIL Image, or numpy array
        confidence_threshold: minimum confidence to accept a detection (default 0.80)

    Returns: list of dicts, one per accepted detection:
        {
          "label":      str,   — class name from COCO_LABELS
          "confidence": float, — confidence score (0.0 to 1.0)
          "box":        dict   — {"x": int, "y": int, "w": int, "h": int}
                                 in pixel coordinates of the original image
        }
    """
    # Build the 4D blob and keep the original image for pixel-space conversion.
    blob, original = preprocess_for_detection(image_source)
    image_height, image_width = original.shape[:2]

    # Feed the blob through the network and run the forward pass.
    net.setInput(blob)
    detections = net.forward()  # shape: (1, 1, N, 7)

    results = []
    # Iterate over the N candidate detections along dimension 2.
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])

        # Drop anything below the requested confidence threshold.
        if confidence < confidence_threshold:
            continue

        # Anti-bug rule #3: always cast the class id to int before indexing,
        # and bounds-check it against the label list.
        class_id = int(detections[0, 0, i, 1])
        if not (0 < class_id < len(COCO_LABELS)):
            continue

        # Anti-bug rule #4: the model emits NORMALISED corner coordinates in
        # [0, 1]. Convert to pixel space FIRST, then derive width/height.
        x = int(detections[0, 0, i, 3] * image_width)
        y = int(detections[0, 0, i, 4] * image_height)
        x2 = int(detections[0, 0, i, 5] * image_width)
        y2 = int(detections[0, 0, i, 6] * image_height)

        # Anti-bug rule #9: clamp to non-negative coordinates.
        x = max(0, x)
        y = max(0, y)
        w = max(0, x2 - x)
        h = max(0, y2 - y)

        results.append({
            "label": COCO_LABELS[class_id],
            "confidence": confidence,
            "box": {"x": x, "y": y, "w": w, "h": h},
        })

    return results


def _color_for_class(label):
    """Generate a deterministic BGR colour for a class label from its index."""
    # Use the class index to seed a repeatable colour so a given class always
    # gets the same box colour across runs.
    index = COCO_LABELS.index(label) if label in COCO_LABELS else 0
    rng = np.random.RandomState(index * 7 + 1)
    color = rng.randint(0, 255, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def annotate_detections(image_source, detections):
    """
    Draw bounding boxes and labels on the original image for all detections.
    Use a distinct color per class (generate from a fixed color map using class index).
    Label format: "cat: 91.3%"
    Box line thickness: 2 pixels.
    Font: cv2.FONT_HERSHEY_SIMPLEX, scale 0.6.
    Returns: annotated BGR numpy array (copy of original, not modified in place).
    """
    # Work on a copy so the source image is never mutated in place.
    original = load_image(image_source)
    annotated = original.copy()

    for det in detections:
        box = det["box"]
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        color = _color_for_class(det["label"])

        # Bounding box rectangle, 2px thick per spec.
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        # Label text e.g. "cat: 91.3%".
        label_text = f"{det['label']}: {det['confidence'] * 100:.1f}%"

        # Place the label just above the box, or just below the top edge if the
        # box starts at the very top of the image.
        text_y = y - 8 if y - 8 > 10 else y + 18
        cv2.putText(
            annotated,
            label_text,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return annotated


def save_detection_result(annotated_image, output_dir, filename="detection_result.png"):
    """
    Save annotated image to output_dir/detection_results/filename.
    Creates directory if it does not exist.
    Returns: full path of saved file as a string.
    """
    target_dir = Path(output_dir) / "detection_results"
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / filename
    cv2.imwrite(str(output_path), annotated_image)
    return str(output_path)
