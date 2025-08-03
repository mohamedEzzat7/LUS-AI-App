"""Utility functions for the LUS‑AI‑App.

This module contains helper functions to load the pre‑trained lung
ultrasound scoring model from Hugging Face, perform inference on
uploaded images, and implement a simple B‑line detection algorithm
using classical computer vision techniques.  The B‑line detector is
not a fully fledged deep learning solution but rather a pragmatic
heuristic that identifies bright vertical artefacts (B‑lines) in
ultrasound images.  This provides a demonstration of how one might
quantify B‑lines when a specialised model is unavailable.
"""

from functools import lru_cache
from typing import Tuple

import numpy as np
from PIL import Image
import cv2
import torch
from transformers import AutoProcessor, AutoModelForImageClassification

# -----------------------------------------------------------------------------
# Model loading
#
# Loading models can be expensive; use an LRU cache so that repeated
# invocations reuse the same objects.  Streamlit's @st.cache_resource
# decorator is ideal for caching in the web app, but using a plain
# Python cache makes this module library‑agnostic and easy to test.

@lru_cache(maxsize=1)
def load_lus_model() -> Tuple[AutoProcessor, AutoModelForImageClassification]:
    """Load the pre‑trained lung ultrasound scoring model.

    Returns
    -------
    Tuple[AutoProcessor, AutoModelForImageClassification]
        A tuple containing the processor and the PyTorch model.

    Notes
    -----
    The model is loaded from Hugging Face using the model ID
    ``hamdan07/UltraSound-Lung``.  If network connectivity is
    unavailable the download may fail; in that case, ensure the
    appropriate files are available in the local cache or specify a
    local model path instead of the remote ID.
    """
    processor = AutoProcessor.from_pretrained("hamdan07/UltraSound-Lung")
    model = AutoModelForImageClassification.from_pretrained("hamdan07/UltraSound-Lung")
    model.eval()
    return processor, model


def predict_lus_score(image: Image.Image) -> Tuple[str, str]:
    """Predict the Lung Ultrasound (LUS) score for a single image.

    This function uses the pre‑trained vision transformer model to
    classify the input image into one of three categories provided by
    the model: ``regular`` (normal), ``pneumonia`` or ``covid``.
    These coarse disease categories are then mapped onto the LUS
    scoring system (0–3), where the score reflects the severity of
    lung aeration loss observed in the image.  Because the model was
    not explicitly trained for LUS scoring, the mapping below is
    heuristic and should not be treated as clinically validated.

    Parameters
    ----------
    image : PIL.Image.Image
        A PIL image in RGB mode.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the predicted LUS score (as a string
        representing an integer between 0 and 3) and a human‑readable
        description of the predicted pattern.
    """
    processor, model = load_lus_model()

    # Preprocess the image for the model
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax(dim=-1).item()
        predicted_label = model.config.id2label[str(predicted_class_id)]

    # Map the model's disease labels to LUS scores.  This mapping is
    # heuristic: ``regular`` is considered normal A‑lines (score 0),
    # ``pneumonia`` corresponds to coalescent B‑lines (score 2) and
    # ``covid`` corresponds to consolidation (score 3).  The
    # intermediate score 1 (moderate B‑lines) is assigned when the
    # model predicts ``regular`` but the B‑line detector finds a small
    # number of lines.
    score_map = {
        "regular": "0",       # Normal pattern
        "pneumonia": "2",    # Coalescent B‑lines
        "covid": "3"        # Consolidation
    }
    # Default description associated with each score
    description_map = {
        "0": "Normal A‑lines",
        "1": "Moderate B‑lines",
        "2": "Coalescent B‑lines",
        "3": "Consolidation"
    }

    # Start with the heuristic mapping
    lus_score = score_map.get(predicted_label, "0")

    # If the model predicts "regular" we further inspect the image for
    # vertical artefacts.  A small number of B‑lines bumps the score
    # from 0 to 1 to capture moderate B‑lines.  This simple rule
    # provides some granularity without a specialised model.
    if predicted_label == "regular":
        b_count = detect_b_lines(image)
        if 1 <= b_count <= 3:
            lus_score = "1"

    return lus_score, description_map[lus_score]


def detect_b_lines(image: Image.Image) -> int:
    """Detect vertical artefacts (B‑lines) in a lung ultrasound image.

    B‑lines appear as bright, vertically oriented artefacts extending
    from the pleural line to the bottom of the ultrasound image.  This
    function uses basic image processing to highlight bright
    structures and applies a Hough transform to count approximate
    vertical lines.

    Parameters
    ----------
    image : PIL.Image.Image
        A PIL image in RGB mode.

    Returns
    -------
    int
        The approximate number of vertical lines detected.  While
        counts may not match clinical definitions precisely, larger
        numbers generally indicate more pronounced B‑line patterns.
    """
    # Convert to grayscale and resize for faster processing
    rgb = np.array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Apply a median blur to reduce speckle noise
    blurred = cv2.medianBlur(gray, 5)
    # Enhance bright structures by thresholding using Otsu's method
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Detect edges using Canny
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    # Use probabilistic Hough transform to detect line segments
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50,
                            minLineLength=30, maxLineGap=10)

    count = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Compute angle of the line: we consider it vertical if the
            # horizontal displacement is small relative to the vertical
            # displacement.  A tolerance angle of 10 degrees (~tan(10°) ≈ 0.176)
            # is used.
            dx = x2 - x1
            dy = y2 - y1
            if abs(dy) > 0 and abs(dx) / abs(dy) < 0.176:  # approx <10° from vertical
                count += 1

    return count