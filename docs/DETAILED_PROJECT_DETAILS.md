# Alzheimer's Disease Detection Project: Detailed Documentation

This document serves as a comprehensive, low-level technical manual for the Alzheimer's Disease Detection project. It details the system architecture, dataset nuances, modular code structure, multi-phase machine learning pipelines, and the integration of a production-ready diagnostic web application.

---

## 1. Project Architecture Overview

The core objective of this project is to build an interpretable, robust machine learning pipeline capable of predicting clinical dementia using both demographic/clinical profiles and neuroimaging-derived structural biomarkers. 

To achieve this, the project is structured chronologically into two primary machine learning phases, ultimately culminating in an interactive web application:
1. **Phase 1 (Clinical Baseline):** Training robust baseline models using easily obtainable clinical variables and cognitive scores.
2. **Phase 2 (Enhanced Biomarkers & Ablation):** Extracting granular volumetric brain metrics from raw MRI scans, training enhanced models, and conducting ablation studies to prove clinical validity without relying heavily on cognitive tests.
3. **Web Application (NeuroScan AI):** Deploying the best-performing models to a responsive, real-time diagnostic dashboard built with FastAPI and Vanilla JS.

---

## 2. Dataset Strategy & Preprocessing

The project relies on the **OASIS-1 (Open Access Series of Imaging Studies)** dataset. The data is logically separated into `data/raw`, `data/processed`, and `data/enhanced_features`.

### 2.1 Demographic & Clinical Data (`data/raw/oasis_cross-sectional.csv`)
- **Demographics:** Age (18-96), M/F (Gender), Educ (Years of Education), SES (Socioeconomic Status).
- **Clinical/Cognitive:** MMSE (Mini-Mental State Examination score, 0-30).
- **Target Variable (CDR):** The Clinical Dementia Rating. This is binarized during preprocessing into `0` (Healthy/Non-Demented) and `1` (Demented/Alzheimer's).

### 2.2 Preprocessing (`src/preprocessor.py`)
- **Missing Value Handling:** The `SES` (Socioeconomic Status) field contains ~5% missing values, which are imputed using the population median. Other rows with missing critical labels are dropped.
- **Encoding:** Gender (`M/F`) is categorically encoded to `0` and `1`.
- **Scaling:** Standard scaling (`StandardScaler`) is strictly applied to continuous numerical variables to ensure models like SVM and KNN compute distances correctly and to stabilize gradient-boosting models.

---

## 3. Exploratory Data Analysis & Clinical Auditing

Before training, extensive EDA (`scripts/data_visualization.py`) and clinical sanity checks (`scripts/pre_training_audit.py`) were performed to guarantee the integrity of our assumptions.

### 3.1 Key Findings
- **Class Imbalance:** Approximately 70% of the dataset is healthy, and 30% is demented. This imbalance dictates our use of Stratified K-Fold CV, F1-scores, and ROC-AUC over simple accuracy.
- **Structural Sanity Checks (`pre_training_audit.py`):**
  - **Hippocampal Atrophy:** Healthy patients average ~3200 mm³, while demented patients show severe atrophy down to ~2500 mm³.
  - **Ventricular Enlargement:** Demented patients exhibit significantly enlarged ventricles (~45,000 mm³ vs ~30,000 mm³ for healthy).
  - **CSF-to-Brain Ratio:** Elevated in dementia patients, mirroring the loss of cortical tissue.

---

## 4. Phase 1: Clinical Baseline Modeling

The Phase 1 pipeline (`scripts/train_all_models.py`) establishes the benchmark utilizing standard clinical features (`Age, M/F, Educ, SES, MMSE, eTIV, nWBV, ASF`).

### 4.1 Methodology
- **Validation:** 80/20 train/test split utilizing 5-fold Stratified Cross-Validation.
- **Algorithm Suite:** Logistic Regression, SVM, Random Forest, XGBoost, Gradient Boosting, KNN, Naive Bayes, AdaBoost.
- **Performance:** **XGBoost** and **Random Forest** dominated the baseline, achieving ~0.94 ROC-AUC. 
- **Interpretability:** SHAP/Feature Importance analysis revealed that the `MMSE` score accounts for over 70% of the model's predictive power, highlighting an over-reliance on a simple cognitive exam.

---

## 5. Phase 2: Enhanced MRI Feature Extraction & Ablation

To build a model that understands the actual physiological changes of Alzheimer's rather than just cognitive decline, Phase 2 (`scripts/train_phase2_enhanced.py`) introduced custom imaging markers.

### 5.1 Imaging Pipeline (`src/imaging/` & `scripts/run_full_oasis1_pipeline.py`)
The pipeline processes raw NIfTI/HDR MRI scans (FSL-segmented) across hundreds of patient sessions.
- **Tissue Compartmentalization:** Calculates absolute volumes of Gray Matter (GM), White Matter (WM), and Cerebrospinal Fluid (CSF).
- **Ratios:** Computes `csf_to_brain_ratio` and `gm_wm_ratio` to quantify cortical shrinkage.
- **ROIs:** Extracts Hippocampal and Ventricular volumes directly from the segmentations.
- **Output:** Combines with Phase 1 data to create `data/enhanced_features/oasis1_full_enhanced_features.csv`.

### 5.2 Ablation Studies (`scripts/ablation_study.py`)
To rigorously test the validity of the structural biomarkers, models are evaluated across three modalities:
1. **Full Mode:** Demographics + Clinical (MMSE) + All structural MRI features. (Highest ROC-AUC: ~0.945).
2. **No-MMSE Mode:** Demographics + Structural MRI features (MMSE dropped). Forces the model to rely on structural atrophy. Result: Maintained a highly robust ROC-AUC of ~0.923, proving structural MRI validity.
3. **Imaging-Only Mode:** Purely MRI structural features (No Educ, No SES, No MMSE). 

---

## 6. Core Module Breakdown (`src/`)

To ensure reproducibility and clean architecture, the core logic is abstracted away from the execution scripts:
- `src/preprocessor.py`: Contains modular classes for imputing, encoding, and scaling tabular data.
- `src/models.py`: Encapsulates the hyperparameter configurations and pipeline definitions for all 8 evaluated ML algorithms.
- `src/utils.py`: Provides standardized metric reporting (Precision, Recall, F1, ROC-AUC), ROC curve plotting, and model artifact saving/loading utilities.
- `src/imaging/`: Contains the logic for parsing and aggregating metrics from 3D NIfTI neuroimaging files.

---

## 7. Web Application Architecture (NeuroScan AI)

The models are operationalized via a production-grade diagnostic dashboard, bridging the gap between machine learning and clinical end-users. 

### 7.1 Backend (`webapp/backend/main.py`)
- **Framework:** FastAPI, chosen for its asynchronous capabilities, speed, and automatic Swagger documentation.
- **State Management:** Upon startup, the backend pre-loads the trained `XGBoost` model artifacts from Phase 1, and the distinct ablation models (Full, No-MMSE, Imaging-Only) from Phase 2. Scalers are also loaded into memory.
- **Endpoints:**
  - `/patients`: Retrieves a list of available session IDs dynamically from the enhanced dataset.
  - `/predict/{mode_key}`: Performs on-the-fly data preprocessing and model inference. It aligns the patient's feature vector with the requested model's exact signature, returning:
    - Prediction (Healthy vs. Demented).
    - Confidence Probabilities.
    - Feature Importance coefficients for Chart.js rendering.

### 7.2 Frontend (`webapp/frontend/`)
- **Tech Stack:** Vanilla JavaScript (`app.js`), HTML5 (`index.html`), and CSS3 (`style.css`).
- **UI/UX:** Implements a modern, premium "glassmorphism" aesthetic with a dark-mode theme, sleek gradients, and micro-animations to create a cutting-edge clinical feel.
- **Dual-Dashboard Integration:** The frontend is designed to compare Phase 1 against Phase 2 models seamlessly.
  - Users select a patient via a dropdown.
  - The application concurrently queries both the Phase 1 and Phase 2 endpoints.
  - It renders a side-by-side comparative UI highlighting differences in model confidence and the shifting importance of features (e.g., how the model shifts reliance from `MMSE` in Phase 1 to `Ventricles` and `Hippocampus` in Phase 2).

---

## 8. Deployment and Execution

The project relies on strict virtual environment isolation. 
- The ML training pipelines run within the project root using standard `requirements.txt`.
- The Web App utilizes a dedicated automation script (`run_webapp.sh`) which bootstraps the backend API via `uvicorn` and serves the frontend over a standard HTTP server, automatically handling Python virtual environment activation (`webapp_venv` / `.venv`) to prevent dependency conflicts.
