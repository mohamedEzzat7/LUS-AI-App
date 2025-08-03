# LUS AI App

**LUS AI App** is a web application built with [Streamlit](https://streamlit.io/) that automatically analyses lung ultrasound (LUS) images. The app performs two main tasks:

1. **LUS Scoring (0–3):** Uses a pre‑trained vision transformer model (from the Hugging Face repository [`hamdan07/UltraSound‑Lung`](https://huggingface.co/hamdan07/UltraSound-Lung)) to classify each image and map the result onto the four‑point LUS score. Scores range from `0` (normal A‑lines) to `3` (consolidation). Because the underlying model was trained on disease categories rather than LUS patterns, the mapping is heuristic and should not be treated as a definitive clinical diagnosis.

2. **B‑Line Detection:** Implements a simple computer‑vision pipeline to detect bright, vertically oriented artefacts known as **B‑lines**. The algorithm converts the uploaded image to grayscale, enhances bright structures, applies edge detection and counts vertical line segments via a Hough transform. The resulting count is converted into qualitative categories such as “No B‑lines” or “Coalescent B‑lines”.

> **Disclaimer:** This application is intended for research and educational purposes only. It is **not** a substitute for professional medical advice or diagnosis. Always consult a qualified clinician when interpreting ultrasound images.

## Features

- **Drag‑and‑drop upload:** Load one or multiple JPEG/PNG ultrasound frames directly in the browser.
- **Automated scoring:** Predicts a LUS score for each image using a transformer model fine‑tuned on lung ultrasound data.
- **B‑line quantification:** Estimates the number of B‑lines and provides a descriptive summary.
- **Interactive results:** Displays the uploaded image alongside the predictions and highlights the interpretive categories.

## Installation

To run the app locally, clone this repository and install the Python dependencies listed in `requirements.txt`. It is recommended to use Python ≥ 3.9.

```bash
git clone https://github.com/mohamedEzzat7/LUS-AI-App.git
cd LUS-AI-App
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first time you run the app it will download the pre‑trained transformer weights from Hugging Face. Ensure your machine has internet connectivity or pre‑download the model into the Hugging Face cache.

## Usage

Launch the Streamlit server by executing:
streamlit run app.py

Your default browser should open automatically displaying the app. Use the file uploader to select lung ultrasound frames. For each image the app will display:

- The original image.
- The predicted LUS score (0–3) and a corresponding description.
- The approximate number of detected B‑lines and a qualitative category.
  
## Project Structure

├── app.py           # Streamlit user interface
├── utils.py         # Model loading and B‑line detection functions
├── requirements.txt # Python dependencies
└── README.md        # Project documentation

Feel free to customise the detection thresholds or extend the model mappings in utils.py to better suit your dataset. To deploy the application on Hugging Face Spaces simply push this repository to a public GitHub repo and follow the Spaces instructions for linking a Streamlit app.

يمكنك نسخ هذا المحتوى ولصقه مباشرة في ملف README.md في مستودع المشروع.
