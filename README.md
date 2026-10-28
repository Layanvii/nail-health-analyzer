# Nail Health Analyzer

A Streamlit computer-vision prototype that lets a user upload an image, isolate a finger area, crop a nail region, and inspect image-derived visual features.

> **Important:** This is an educational image-processing prototype, not a medical device. Its color-based estimates and heuristic scores are not clinically validated and must not be used to diagnose anemia, iron deficiency, hypoxemia, or any other condition. They do not replace laboratory tests or advice from a qualified clinician.

## What it does

- Uploads an image through a Streamlit interface.
- Applies OpenCV-based color thresholding and morphology to attempt background removal.
- Lets the user crop the region of interest with streamlit-cropper.
- Computes image features such as average RGB values, brightness, saturation, and simple visual indicators.
- Presents heuristic outputs for exploration and demonstration.

## Tech stack

- **Python**
- **Streamlit** — interactive web interface
- **OpenCV** — image processing
- **NumPy** — numerical operations
- **Pillow** — image loading
- **Matplotlib** — plotting utilities
- **streamlit-cropper** — interactive crop selection

## Getting started

### Requirements

- Python 3.10+ recommended
- pip

### Install

    git clone https://github.com/Layanvii/nail-health-analyzer.git
    cd nail-health-analyzer
    python -m venv .venv

Activate the virtual environment:

**Windows PowerShell**

    .venv\Scripts\Activate.ps1

**macOS / Linux**

    source .venv/bin/activate

Install dependencies and launch the app:

    pip install -r requirements.txt
    streamlit run app.py

Streamlit will print a local URL in the terminal. Open that URL in your browser.

## Project structure

    nail-health-analyzer/
    ├── app.py
    ├── requirements.txt
    └── README.md

## Limitations and responsible use

- Results depend on lighting, camera settings, skin tone, nail polish, image quality, cropping, and background-removal performance.
- The current thresholds and scores are hand-crafted heuristics; they are not trained or validated against clinical measurements.
- A photograph cannot reliably determine hemoglobin concentration, blood oxygen saturation (SpO₂), or iron status.
- Do not use the outputs to make health decisions. For concerns, consult a qualified healthcare professional and use appropriate validated tests.

## Future improvements

- [ ] Add representative sample images and interface screenshots.
- [ ] Add automated tests for image-processing functions and edge cases.
- [ ] Separate visual feature extraction from the Streamlit UI.
- [ ] Evaluate image-processing robustness across varied lighting and image conditions.
- [ ] Remove or clearly label any health-related numerical estimates that are not supported by validated data.

## Author

**Layan** — [GitHub profile](https://github.com/Layanvii)
