# Alzheimer's Disease Detection from Brain MRI Data

**By:** Jainil Rana, Mansi Gupta, Mohit Barade  
**Course:** CMPE-257 Machine Learning by Prof. Zara Hajihashemi

---

## 📌 Problem Description
Alzheimer's is a neurodegenerative disease that results in memory loss, cognitive impairment, and loss of independence. One of the key challenges in managing Alzheimer's is that structural changes in the brain, such as tissue shrinkage and ventricular enlargement, begin years before noticeable clinical symptoms appear. While MRI scans capture these early changes, reliably identifying them at scale is highly complex.

Our project focuses on building robust, **interpretable machine learning models** to detect Alzheimer's disease. Instead of treating the models as black boxes, we aim to uncover how distinct clinical metrics and structural imaging features contribute to predictions, ensuring clinical utility and transparency in the decision-making process.

The project is structured into **two distinct phases**:
- **Phase 1: Clinical Baseline Modeling** - Applying interpretable ML algorithms (Random Forest, SVM, XGBoost, etc.) to demographic and tabular clinical data to establish a robust baseline.
- **Phase 2: Enhanced Imaging Biomarkers & Ablation** - Extracting handcrafted structural imaging biomarkers (tissue composition and regional volumes) from raw MRI scans, training enhanced models, and conducting ablation studies to test clinical robustness.

---

## 📊 Primary Dataset
**OASIS-1:** Open Access Series of Imaging Studies ([Link](https://www.oasis-brains.org/))

### Dataset Description
- **Number of Subjects:** ~416 individuals (cross-sectional)
- **Age Range:** 18–96 years
- **Data Types:**
  - Structural T1-weighted MRI scans (3D NIfTI/HDR format)
  - Clinical and demographic data in tabular CSV format
- **Key Features:**
  - Demographics (Age, Sex, Education, Socioeconomic Status)
  - Cognitive assessment scores (e.g., MMSE)
  - Precomputed gross brain volume measures (e.g., eTIV, normalized whole brain volume - nWBV)
- **Target Variable:** Clinical Dementia Rating (CDR) / Dementia Status

---

## ⚙️ Methodology & Pipeline

### Phase 1: Baseline Interpretable Machine Learning
- **Preprocessing:** Handling missing data, standardizing numerical features, and filtering for valid clinical labels.
- **Model Training:** Training 8 classic ML models (Random Forest, Logistic Regression, SVM, XGBoost, Gradient Boosting, KNN, Naive Bayes, AdaBoost) using k-fold cross-validation.
- **Interpretability:** Evaluating feature importance coefficients to identify the strongest baseline predictors (like MMSE and Age).
- **Goal:** Establish a reliable, interpretable benchmark against which we can measure improvements gained from adding neuroimaging features.

### Phase 2: Enhanced MRI Feature Extraction & Ablation
- **Imaging Feature Extraction:** We developed a full-scale pipeline (`run_full_oasis1_pipeline.py`) to process raw FSL-segmented brain MRIs across all dataset discs.
- **Handcrafted Biomarkers:** 
  - **Tissue Composition:** Extracted Gray Matter (GM), White Matter (WM), and Cerebrospinal Fluid (CSF) volumes, deriving fractions and ratios (e.g., CSF-to-brain ratio).
  - **Regional ROIs:** Extracted specific neurodegenerative indicators like Hippocampal volume and Ventricular enlargement.
- **Enhanced Model Training:** We integrate these rich imaging biomarkers into our Phase 1 models to see how structural patterns improve predictive accuracy (`train_phase2_enhanced.py`).
- **Ablation Studies:** To ensure our model relies on genuine brain structural changes rather than just direct cognitive tests, we perform ablation tests:
  - **Without MMSE:** Testing how models perform without the leading cognitive indicator.
  - **Imaging-Only:** Testing models relying purely on extracted brain structure biomarkers (GM/WM/CSF/Regional) and basic demographics.

---

## 🚀 Key Results & Project Structure

The codebase is organized into modular scripts for end-to-end execution:
- `src/` - Core Python modules for preprocessing (`preprocessor.py`), model definitions (`models.py`), and imaging feature extraction (`imaging/`).
- `scripts/` - Execution pipelines:
  - `train_all_models.py` - Runs Phase 1 baseline training.
  - `run_full_oasis1_pipeline.py` - Parses 400+ session MRIs, merges with clinical data, and validates clinical plausibility.
  - `train_phase2_enhanced.py` - Runs the Phase 2 training suite and ablation studies.
  - `ablation_study.py` & `evaluate_all_models.py` - Evaluation utilities.
- `models/` - Saved model artifacts for Phase 1 and Phase 2 variations (Full, No-MMSE, Imaging-Only).

---

## ⚠️ Addressed Challenges
- **Class Imbalance:** Handled via robust metrics (F1-score, Precision-Recall AUC) and cross-validation to prevent overfitting on the majority class.
- **Confounding Impact of Age:** Normal aging mimics early Alzheimer’s structural changes. We counteract this by maintaining age as a controlled variable and extracting localized ROIs (hippocampus) less universally affected by normal aging alone.
- **MRI Heterogeneity:** Variations in MRI resolution are mitigated by utilizing standardized FSL segmentation masks and robust normalization (like eTIV scaling).

---

## 📈 Evaluation Metrics
We rely on rigorous evaluation, utilizing stratified k-fold cross-validation to ensure reliable generalizability. The primary metrics used are:
- **F1-Score & Precision-Recall AUC:** Critical for evaluating the model on the imbalanced clinical dataset.
- **ROC-AUC:** Assesses the model's capability to correctly separate demented vs. non-demented patients.
- **Accuracy:** Tracked alongside detailed confusion matrices and classification reports.

---

## 📚 References
- Marcus, D. S., et al., “Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI Data in Young, Middle Aged, Nondemented, and Demented Older Adults,” *Journal of Cognitive Neuroscience*, 2007.
- OASIS Brain Project. [https://www.oasis-brains.org/](https://www.oasis-brains.org/)
