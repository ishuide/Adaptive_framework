# Adaptive Ensemble Framework for AI-Generated Image Detection

## Overview
This repository contains the development of an adaptive ensemble pipeline designed to distinguish AI-generated images from real ones. Developed over a 3-month period, the project evolved from a simple four-detector baseline into a massive 15-detector pipeline applying 44 image transformations per sample. The core concept relies on extracting robust forensic metadata from heavily transformed variants of an image to build a meta-model ensemble.

## Development Timeline & Methodology

### Month 1 — Foundation (May)
The project began by surveying existing detectors and establishing a baseline pipeline.
*   **Initial Pipeline:** Built around 4 base signal detectors (ViT, Error Level Analysis, FFT, and Noise Analysis).
*   **Transformations:** Defined a core set of ~9 transformation types (JPEG compression, resizing, cropping, blurring, noise, brightness, contrast, sharpening, screenshots).
*   **Data Preparation:** Enforced a strict ground-truth convention where real images (and their transformed derivatives) are always labeled `0`. Uploaded our working dataset to Kaggle and identified existing HuggingFace models (e.g., dima806) as benchmarks.

### Month 2 — Scaling the Ensemble (June)
The focus shifted to addressing false positives/negatives (FP/FN) and scaling the ensemble.
*   **Pipeline Formalization:** The standard flow was locked in: `Transform` → `Run Base Detectors` → `Build Metadata` → `Train Meta-Model` → `Save weights`.
*   **Massive Scaling (notebooks/01_v3_baseline_15_detectors.ipynb):** Expanded the pipeline to **15 detectors** (3 GPU: SigLIP/ViT/CLIP + 12 CPU forensic detectors) and **44 unique transformations**. 
*   **Ensemble Strategy:** Moved to a 3-model soft-voting ensemble (XGBoost, LightGBM, RandomForest) replacing the fixed-threshold approach with a more dynamic mechanism.
*   **Tuning (notebooks/02_v3_optimized_ensemble.ipynb):** A genuine tuning pass on the v3 architecture to refine features and stabilize the ensemble without data leakage.

### Month 3 — Prompt Upgrades, Backend, and the Generalization Wall
The final month brought advanced feature extraction and a critical evaluation of the model's real-world limitations.
*   **CLIP-Enhanced Upgrade (notebooks/03_v4_clip_enhanced.ipynb):** Added a CLIP-Enhanced Multi-Prompt detector. On an in-domain held-out test set of 116 untransformed images, this achieved 98.28% accuracy.
*   **The Generalization Wall (Real-World Test):** Recognizing the difference between constrained-domain testing and the wild, a live interactive test was run on a genuine WhatsApp screenshot (documented in cell 38 of the v3 baseline). The transformation sweep returned 21 "AI" votes vs 23 "Real" votes—a near coin-flip. This exposed the extreme difficulty of extracting usable metadata from heavily compressed messaging-app images, demonstrating where the single-pass ensemble struggles to generalize.
*   **Parallel Streaming Experiment (notebooks/04_inference_pipeline.ipynb):** Initially named `final.ipynb`, this notebook represents a separate, parallel experiment testing a dual-stream end-to-end model trained on 96k streamed images, independent of the main 15-detector ensemble.

## Dataset
The dataset used for training and evaluation is hosted on Kaggle:  
**[AI vs Real Image Dataset on Kaggle](https://www.kaggle.com/datasets/ishuide/real-detector-dataset-v2)**

> Replace the link above with your actual Kaggle dataset URL if different.

## Current Project Status
*   **Backend:** Completed. 
*   **Frontend / UI:** A Streamlit-based UI (`app.py`) is included for interactive single-image detection.
*   **Next Steps:** Developing a secondary, lighter-weight pipeline dedicated specifically to correcting the FP/FN edge cases (like WhatsApp compressions) by analyzing metadata extraction patterns.

## Repository Structure
*   `notebooks/01_v3_baseline_15_detectors.ipynb`: The scaled 15-detector / 44-transformation pipeline.
*   `notebooks/02_v3_optimized_ensemble.ipynb`: The tuning pass for the v3 ensemble.
*   `notebooks/03_v4_clip_enhanced.ipynb`: Introduction of the CLIP multi-prompt detector.
*   `notebooks/04_inference_pipeline.ipynb`: The parallel dual-stream end-to-end model experiment.
*   `app.py`: Backend + UI implementation for interactive image detection.
*   `requirements.txt`: Python dependencies needed to run the project.
