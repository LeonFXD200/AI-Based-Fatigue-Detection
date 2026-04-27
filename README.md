# Driver Fatigue Detection — Signal Modality Comparison

BSc Computing dissertation by Leon Dupree.
**Evaluating Signal-Based Approaches for AI Driver Fatigue Detection.**

This repository contains the code, training notebooks, and live demo for a
comparative study of four fatigue detection modalities: **eye behaviour**,
**facial expression**, **voice patterns**, and **steering behaviour**.

## Project structure

```
fatigue-detection/
├── notebooks/        # training notebooks (one per modality)
├── src/              # clean, importable Python modules
│   ├── eye/
│   ├── face/
│   ├── voice/
│   └── steering/
├── models/           # trained model weights (committed)
├── data/             # datasets (gitignored — download separately)
├── tests/            # unit tests
├── app.py            # Gradio demo app (built last)
└── requirements.txt
```

## Datasets

| Modality | Dataset | Source |
|----------|---------|--------|
| Eye | MRL Eye Dataset | Kaggle / MRL VŠB-TUO |
| Face | Drowsiness Detection (Kaggle) | Kaggle |
| Voice | Self-recorded pilot | (single subject — see dissertation) |
| Steering | UAH-DriveSet | University of Alcalá |

Datasets must be downloaded into the `data/` folder before training. See
each notebook's setup cell for the expected folder structure.

## Methodology summary

A consistent ML approach is used across modalities where possible. Face,
voice, and steering use a small MLP trained on per-second engineered
features. Eye uses a small CNN on raw eye-region image crops, justified by
the difference in input modality (raw pixels vs. derived signals).

All models are evaluated using accuracy, precision, recall, F1-score, and
confusion matrix. Train/validation/test splits are subject-level, not
random frame-level, to ensure honest cross-subject generalisation
estimates.

## Live demo

A Gradio app hosting all four trained models is deployed on Hugging Face
Spaces. Users can upload video clips and rate the predictions, generating
real-world cross-subject accuracy data for the dissertation.

[Demo link to be added once deployed]

## Quick start

```bash
git clone https://github.com/<your-username>/fatigue-detection.git
cd fatigue-detection
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter lab notebooks/
```
