"""Streamlit application for Lung Ultrasound (LUS) scoring and B‑line detection.

This app allows users to upload lung ultrasound images (PNG or JPEG) and
automatically generates a predicted LUS score (0–3) alongside a
count of detected B‑lines.  The scoring leverages a pre‑trained
transformer model to classify each image and a simple computer vision
routine to estimate the number of vertical artefacts.  Results are
displayed with descriptive text and a disclaimer to emphasise that
outputs are for demonstration purposes only and should not be used
for clinical decision making without expert review.
"""

import streamlit as st
from PIL import Image
import numpy as np

import utils

# Configure the Streamlit page
st.set_page_config(
    page_title="LUS AI App: Ultrasound Lung Scoring and B‑Line Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🩺 Lung Ultrasound AI")

st.markdown(
    """
    Welcome to the **LUS AI App**, an interactive tool that analyses lung
    ultrasound images.  Upload one or more ultrasound frames below and the
    app will estimate the **LUS score** (ranging from 0 to 3) and count
    visible **B‑lines**, which are vertical artefacts associated with
    pulmonary conditions.  Please note that this tool is intended for
    educational and research purposes; **it does not replace a professional
    medical assessment**.
    """
)

uploaded_files = st.file_uploader(
    "Upload one or more lung ultrasound images (JPEG or PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# Prepare a map from B‑line count to descriptive categories.  These
# categories are heuristic; adjust thresholds as appropriate for your
# use case.
bline_description_map = {
    0: "No B‑lines detected",
    1: "Few B‑lines detected",
    2: "Multiple B‑lines detected",
    3: "Coalescent B‑lines detected"
}

def classify_bline_count(count: int) -> str:
    """Return a qualitative description based on the number of B‑lines."""
    if count == 0:
        return bline_description_map[0]
    elif 1 <= count <= 2:
        return bline_description_map[1]
    elif 3 <= count <= 5:
        return bline_description_map[2]
    else:
        return bline_description_map[3]


if uploaded_files:
    # Display results for each uploaded image
    for uploaded_file in uploaded_files:
        st.markdown("---")
        filename = uploaded_file.name
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Could not process {filename}: {e}")
            continue

        st.subheader(f"Image: {filename}")
        st.image(image, use_column_width=True, caption=filename)

        # Predict LUS score
        with st.spinner("Predicting LUS score..."):
            score, score_desc = utils.predict_lus_score(image)
        st.write(f"**Predicted LUS Score:** {score}  ")
        st.write(f"**Interpretation:** {score_desc}")

        # Detect B‑lines
        with st.spinner("Detecting B‑lines..."):
            b_count = utils.detect_b_lines(image)
        b_desc = classify_bline_count(b_count)
        st.write(f"**Number of B‑lines detected:** {b_count}")
        st.write(f"**B‑line Interpretation:** {b_desc}")

        st.warning(
            "⚠️ These predictions are AI‑generated and for exploratory use only."
            " Always consult a qualified clinician for medical interpretation."
        )
else:
    st.info("Please upload at least one image to begin analysis.")