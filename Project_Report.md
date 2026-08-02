# FINAL REPORT
**Project:** Adaptive Ensemble Framework for AI-Generated Image Detection

## 1. Overview
This report brings together the discussions and decisions recorded across the project's meetings to describe, in a single narrative, how the AI-generated image detection framework has been conceived, built, and refined. The project centers on an adaptive ensemble pipeline that distinguishes AI-generated images from real ones, and the account below traces its progress from the initial concept through data preparation, model design, implementation, and the ongoing testing and expansion work.

## 2. Origins and Pipeline Concept
The project began with a review of detectors already available in the market and a discussion of how an ensemble-based approach could be used to build a more adaptive detection framework. Team members researched ensemble modeling techniques and alternative frameworks. The team settled on a core pipeline: a set of images is put through a series of transformations, each of the resulting variants is passed through several base transformers, the outputs are collected into a metadata set, and that metadata is used to train a meta-model (an ensemble) which is then saved for later use.

## 3. Data Preparation and Transformations
With the pipeline concept agreed upon, both AI-generated and real images were put through a common set of transformations. Initially, these included re-encoding to different JPEG qualities, resizing, cropping, Gaussian blurring, noise addition, brightness/contrast changes, sharpening, and simulated screenshots. Every transformed sample was recorded against a consistent metadata schema capturing the sample identifier, its true label, the transformation applied, and the outputs of each detector. The first concrete step was standardizing every image to a common file format (JPG or PNG).

Later, the transformation set was expanded to include six additional advanced types of image transformations (totaling 44) to broaden the range of distortions the model would be tested against, alongside a parallel effort to gather more test images from the internet.

## 4. Detectors and Model Research
Four base signal detectors were adopted early on as core feature extractors: 
1. Vision Transformer (ViT)
2. Error Level Analysis (ELA)
3. Frequency-Domain (FFT) Analysis
4. Noise Analysis

The working dataset was uploaded to Kaggle. Over time, three additional GPU-based detectors (SigLIP and CLIP) were added to the pipeline to significantly strengthen detection capability, scaling the architecture up to 15 distinct detectors.

## 5. Ground Truth Convention and Ensemble Voting
To avoid inconsistency, the team fixed a clear ground-truth convention: **real images are always labeled as 0**, and this label is carried through to any transformed derivative of a real image. The ensemble approach took three base models (XGBoost, LightGBM, RandomForest), combined their predictions through a soft-voting mechanism, and treated the majority result as the final verdict. 

## 6. Handling False Positives and False Negatives
Recognizing that no single-pass ensemble would be perfect, the team identified the need for a second, lighter-weight pipeline dedicated to correcting false positives (FP) and false negatives (FN). The plan was to focus on metadata extraction, storing FP and FN cases in separate folders to feed into this second-level pipeline. Two ongoing challenges remain: extracting usable metadata from heavily compressed messaging-app screenshots (like WhatsApp/Telegram) and storing FP/FN cases in real time.

## 7. Implementation: Backend and Frontend
The backend implementation was completed successfully. Work is currently planned on designing and building the project's user interface, so that the detection framework can eventually be used through a proper front-facing application rather than purely through backend processing.

## 8. Detailed Development Timeline

### Month 1 — Foundation (May 18 - May 28)
*   **MOM3 (18/05):** Scoped the ensemble approach, surveyed existing detectors, split research among the team.
*   **MOM5 (26/05):** First working pipeline established with 4 detectors (ViT, ELA, FFT, Noise), ~9 transformation types, and defined metadata schema.
*   **MOM6 (27/05):** Dataset uploaded to Kaggle; found 3 candidate HuggingFace detector models for benchmarking.
*   **MOM7 (28/05):** Standardized image format, flagged the need for dynamic thresholding over a fixed one, and established a single-model baseline.

### Month 2 — Scaling the Ensemble (June)
*   **MOM9 (02/06):** Identified the FP/FN correction gap, proposed a second-tier pipeline, and scoped metadata extraction for compressed messaging-app images.
*   **MOM10:** Fixed convention (Real=0 enforced throughout), established the 3-model soft-voting ensemble, and ran six active detectors.
*   **MOM4 (21/06):** Pipeline flow formalized (Transform → Base Detectors → Metadata → Train Meta-Model → Save).
*   **Massive Scaling:** At this point, the ensemble was scaled to **15 detectors** (3 GPU: SigLIP/ViT/CLIP + 12 CPU forensic detectors) and **44 transformations**. A genuine tuning pass was performed to stabilize features without data leakage.

### Month 3 — Prompt Upgrades & The Generalization Wall
*   **CLIP Upgrade:** Added a CLIP-Enhanced Multi-Prompt detector. While this achieved an impressive 98.28% accuracy on an in-domain held-out test set of 116 untransformed images, it highlighted the difference between constrained testing and real-world application.
*   **The Generalization Wall:** A live test was run on a genuine WhatsApp screenshot. Across the 44-transformation sweep, the ensemble returned 21 "AI" votes vs 23 "Real" votes—a near coin-flip on a genuine photo. This empirically demonstrated the extreme difficulty of extracting usable metadata from heavily compressed messaging-app images and exposed the zero-shot generalization limits of a static detector.
*   **Parallel Streaming Experiment:** A final, separate dual-stream end-to-end model trained on 96k streamed images was run as a parallel experiment independent of the main 15-detector ensemble.
