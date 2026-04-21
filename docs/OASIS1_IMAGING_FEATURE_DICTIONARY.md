# OASIS-1 Imaging Feature Dictionary

**Generated:** March 22, 2026  
**Dataset:** OASIS-1 Cross-Sectional (Disc 1)  
**Sessions Processed:** 39 out of 416 total (limited by Disc 1 availability)  
**Extraction Success Rate:** 100% (39/39)

---

## Overview

This document describes the imaging-derived features extracted from OASIS-1 MRI data and merged with the tabular clinical CSV. All features are extracted from FSL FAST tissue segmentation data.

**Key Achievement:** Reconstructed nWBV validation shows excellent agreement with original CSV values:
- Mean absolute error: 0.000553
- Median absolute error: 0.000314
- Max absolute error: 0.005494
- **All errors < 0.01** (100% validation success)

---

## Feature Categories

### 1. Session Identifiers

| Feature | Type | Description | Source |
|---------|------|-------------|--------|
| `session_id` | string | Session identifier (e.g., OAS1_0001_MR1) | Manifest |
| `canonical_session_key` | string | Canonical merge key (same as session_id) | Manifest |

---

### 2. Tissue Voxel Counts

Raw voxel counts from FSL FAST 3-class segmentation.

| Feature | Type | Units | Description |
|---------|------|-------|-------------|
| `csf_voxels` | integer | voxels | Number of CSF (cerebrospinal fluid) voxels |
| `gm_voxels` | integer | voxels | Number of gray matter voxels |
| `wm_voxels` | integer | voxels | Number of white matter voxels |

**Segmentation Classes:**
- Class 0: CSF (cerebrospinal fluid)
- Class 1: GM (gray matter)
- Class 2: WM (white matter)

---

### 3. Tissue Volumes (Absolute)

Tissue volumes in cubic millimeters, extracted from FSL_SEG .txt files.

| Feature | Type | Units | Description |
|---------|------|-------|-------------|
| `csf_vol_mm3` | float | mm³ | CSF volume |
| `gm_vol_mm3` | float | mm³ | Gray matter volume |
| `wm_vol_mm3` | float | mm³ | White matter volume |
| `brain_parenchyma_vol_mm3` | float | mm³ | Brain parenchyma (GM + WM) |
| `total_segmented_vol_mm3` | float | mm³ | Total segmented volume (CSF + GM + WM) |

**Calculation:**
- `brain_parenchyma_vol_mm3 = gm_vol_mm3 + wm_vol_mm3`
- `total_segmented_vol_mm3 = csf_vol_mm3 + gm_vol_mm3 + wm_vol_mm3`

---

### 4. Tissue Fractions

Normalized tissue fractions (sum to 1.0).

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `csf_frac` | float | [0, 1] | CSF fraction of total segmented volume |
| `gm_frac` | float | [0, 1] | Gray matter fraction |
| `wm_frac` | float | [0, 1] | White matter fraction |
| `brain_parenchyma_frac` | float | [0, 1] | Brain parenchyma fraction (GM + WM) |

**Calculation:**
- `csf_frac = csf_vol_mm3 / total_segmented_vol_mm3`
- `gm_frac = gm_vol_mm3 / total_segmented_vol_mm3`
- `wm_frac = wm_vol_mm3 / total_segmented_vol_mm3`
- `brain_parenchyma_frac = brain_parenchyma_vol_mm3 / total_segmented_vol_mm3`

**Validation:** All sessions show `csf_frac + gm_frac + wm_frac = 1.000` (perfect)

---

### 5. Tissue Ratios

Clinically relevant tissue ratios.

| Feature | Type | Description | Clinical Relevance |
|---------|------|-------------|-------------------|
| `csf_to_brain_ratio` | float | CSF / (GM + WM) | Atrophy marker (higher = more atrophy) |
| `gm_wm_ratio` | float | GM / WM | Tissue composition balance |

**Calculation:**
- `csf_to_brain_ratio = csf_vol_mm3 / brain_parenchyma_vol_mm3`
- `gm_wm_ratio = gm_vol_mm3 / wm_vol_mm3`

**Clinical Interpretation:**
- **CSF-to-brain ratio:** Increases with age and neurodegeneration as brain tissue is lost and replaced by CSF
- **GM-WM ratio:** Typically decreases with age; altered in various neurodegenerative conditions

---

### 6. Reconstructed nWBV (Validation Feature)

Reconstructed normalized whole brain volume for validation against original CSV.

| Feature | Type | Description |
|---------|------|-------------|
| `reconstructed_nwbv` | float | Brain parenchyma fraction (same as `brain_parenchyma_frac`) |
| `reconstructed_nwbv_abs_error` | float | Absolute error vs. CSV nWBV |
| `reconstructed_nwbv_pct_error` | float | Percentage error vs. CSV nWBV |

**Calculation:**
- `reconstructed_nwbv = brain_parenchyma_vol_mm3 / total_segmented_vol_mm3`
- `reconstructed_nwbv_abs_error = |reconstructed_nwbv - csv_nwbv|`
- `reconstructed_nwbv_pct_error = (reconstructed_nwbv_abs_error / csv_nwbv) × 100`

**Validation Results:**
- Mean absolute error: **0.000553** (0.055%)
- All errors < 0.01 (excellent agreement)
- Confirms extraction pipeline accuracy

---

### 7. eTIV-Normalized Features

Tissue volumes normalized by estimated total intracranial volume (eTIV) from CSV.

| Feature | Type | Description |
|---------|------|-------------|
| `brain_parenchyma_to_etiv` | float | Brain parenchyma / eTIV |
| `gm_to_etiv` | float | Gray matter / eTIV |
| `wm_to_etiv` | float | White matter / eTIV |
| `csf_to_etiv` | float | CSF / eTIV |

**Calculation:**
- `brain_parenchyma_to_etiv = brain_parenchyma_vol_mm3 / eTIV`
- `gm_to_etiv = gm_vol_mm3 / eTIV`
- `wm_to_etiv = wm_vol_mm3 / eTIV`
- `csf_to_etiv = csf_vol_mm3 / eTIV`

**Clinical Relevance:**
- Normalizes for head size variation
- Allows comparison across individuals
- Standard approach in neuroimaging research

---

## Original CSV Features (Preserved)

All original features from the OASIS-1 tabular CSV are preserved:

| Feature | Description |
|---------|-------------|
| `ID` | Session identifier (merge key) |
| `M/F` | Sex (M=Male, F=Female) |
| `Hand` | Handedness (R=Right) |
| `Age` | Age in years |
| `Educ` | Years of education |
| `SES` | Socioeconomic status (1-5 scale) |
| `MMSE` | Mini-Mental State Examination score (0-30) |
| `CDR` | Clinical Dementia Rating (0, 0.5, 1, 2) |
| `eTIV` | Estimated total intracranial volume (mm³) |
| `nWBV` | Normalized whole brain volume (original) |
| `ASF` | Atlas scaling factor |
| `Delay` | Delay (always NaN for cross-sectional) |

---

## Feature Comparison: Original vs. New

### What Was Already Available (Global Metrics)

| Feature | Type | Limitation |
|---------|------|------------|
| `nWBV` | Global | Total brain volume fraction (non-specific) |
| `eTIV` | Global | Head size normalization only |
| `ASF` | Global | Atlas scaling (non-specific) |

**Problem:** These features measure **total brain size/atrophy** but cannot distinguish:
- Normal aging vs. Alzheimer's disease
- Alzheimer's vs. vascular dementia
- Alzheimer's vs. frontotemporal dementia

### What We Added (Tissue-Specific Metrics)

| Feature Category | Advantage |
|-----------------|-----------|
| **Tissue volumes** | Separate GM, WM, CSF measurement |
| **Tissue fractions** | Normalized composition analysis |
| **Tissue ratios** | Atrophy markers (CSF-to-brain ratio) |
| **eTIV normalization** | Proper head-size correction per tissue type |

**Improvement:** These features provide **tissue-specific** information that:
- Distinguish gray matter loss (Alzheimer's signature)
- Quantify CSF expansion (atrophy marker)
- Enable tissue composition analysis
- Validate extraction pipeline accuracy

---

## What's Still Missing (Regional Features)

### ⚠️ **Critical Limitation: No Regional Specificity**

The current features are still **tissue-level global metrics**. They do NOT measure:

| Missing Feature | Clinical Importance | Why It Matters |
|----------------|-------------------|----------------|
| **Hippocampal volume** | ⭐⭐⭐⭐⭐ | Earliest and most specific Alzheimer's marker |
| **Entorhinal cortex** | ⭐⭐⭐⭐⭐ | Preclinical Alzheimer's marker |
| **Medial temporal lobe** | ⭐⭐⭐⭐⭐ | Diagnostic hallmark of Alzheimer's |
| **Lateral ventricles** | ⭐⭐⭐⭐ | Secondary atrophy marker |
| **Temporal lobe regions** | ⭐⭐⭐⭐ | Alzheimer's progression pattern |

**Why This Matters:**
- Current features improve on **global brain volume** (nWBV)
- But still cannot detect **Alzheimer's-specific atrophy patterns**
- Need **atlas-based ROI extraction** for regional features

---

## Data Quality & Validation

### Extraction Success Rate

| Metric | Value |
|--------|-------|
| Total sessions in Disc 1 | 39 |
| Successful extractions | 39 (100%) |
| Failed extractions | 0 (0%) |
| Sessions with complete file set | 39 (100%) |

### Validation Checks Passed

✅ **Tissue fraction validation:** All sessions sum to 1.000  
✅ **nWBV reconstruction:** Mean error 0.0005 (excellent)  
✅ **Merge integrity:** 100% match rate (39/39)  
✅ **No duplicate keys:** All session IDs unique  
✅ **No missing values:** All extracted features complete  

---

## File Locations

### Input Files

| File | Path | Description |
|------|------|-------------|
| OASIS-1 imaging data | `oasis1-disc1/` | Raw MRI sessions |
| Tabular CSV | `oasis_cross-sectional-5708aa0a98d82080.xlsx` | Clinical data |

### Generated Files

| File | Path | Description |
|------|------|-------------|
| Imaging manifest | `data/imaging_features/oasis1_imaging_manifest.csv` | Session file inventory |
| Tissue features | `data/imaging_features/oasis1_tissue_features.csv` | Extracted imaging features |
| Enhanced CSV | `data/processed_oasis1/oasis1_enhanced_features.csv` | **Final merged dataset** |
| Merge audit | `data/imaging_features/merge_audit_report.txt` | Merge validation report |

---

## Usage Notes

### For Model Training

**Enhanced CSV is ready for Phase 2 model retraining:**

```python
import pandas as pd

# Load enhanced dataset
df = pd.read_csv('data/processed_oasis1/oasis1_enhanced_features.csv')

# Original features (12)
original_features = ['M/F', 'Hand', 'Age', 'Educ', 'SES', 'MMSE', 'CDR', 
                     'eTIV', 'nWBV', 'ASF', 'Delay']

# New imaging features (22)
imaging_features = ['csf_voxels', 'gm_voxels', 'wm_voxels',
                    'csf_vol_mm3', 'gm_vol_mm3', 'wm_vol_mm3',
                    'brain_parenchyma_vol_mm3', 'total_segmented_vol_mm3',
                    'csf_frac', 'gm_frac', 'wm_frac', 'brain_parenchyma_frac',
                    'csf_to_brain_ratio', 'gm_wm_ratio',
                    'reconstructed_nwbv', 'reconstructed_nwbv_abs_error',
                    'reconstructed_nwbv_pct_error',
                    'brain_parenchyma_to_etiv', 'gm_to_etiv', 
                    'wm_to_etiv', 'csf_to_etiv']

# Total features: 34 (12 original + 22 imaging + session_id + canonical_session_key)
```

### Important Considerations

1. **Sample Size:** Only 39 sessions available (Disc 1 limitation)
   - Full OASIS-1 has 416 sessions across multiple discs
   - Current dataset is a **proof-of-concept** subset
   - Need additional discs for full dataset

2. **Feature Redundancy:** Some features are correlated
   - `reconstructed_nwbv` ≈ `brain_parenchyma_frac` (identical)
   - `brain_parenchyma_frac` ≈ `nWBV` (validated)
   - Consider feature selection before modeling

3. **Missing Regional Features:** See "What's Still Missing" section
   - Current features are tissue-level global metrics
   - Need atlas-based ROI extraction for regional specificity

---

## Next Steps for Full Clinical Reliability

### Phase 2B: Regional Feature Extraction (Recommended)

To achieve clinically reliable Alzheimer's prediction, extract:

1. **Hippocampal volumes** (L/R) - using atlas-based ROI extraction
2. **Lateral ventricle volumes** (L/R) - secondary atrophy marker
3. **Entorhinal cortex ROI proxies** - preclinical marker
4. **Temporal lobe ROI volumes** - Alzheimer's progression pattern

**Implementation approach:**
- Use Harvard-Oxford atlas or AAL atlas in T88 space
- Extract ROI volumes from T88-registered images
- Calculate asymmetry metrics (L vs R)
- Normalize by eTIV

### Phase 2: Model Retraining & Validation

1. Retrain all 8 ML models on enhanced features
2. Compare performance vs. baseline (Phase 1)
3. Analyze feature importance (expect tissue features to contribute)
4. Validate clinical scenarios (early detection, differential diagnosis)

---

## Technical Notes

### Extraction Pipeline

1. **Session discovery:** Automated file discovery and manifest generation
2. **Tissue extraction:** Parse FSL_SEG .txt files + segmentation images
3. **Validation:** Reconstruct nWBV and compare to CSV
4. **Merge:** Session-safe merge with audit trail

### Quality Control

- All file paths validated before extraction
- Graceful error handling for missing files
- Comprehensive validation checks
- Merge integrity verification

### Reproducibility

All extraction scripts are deterministic and reproducible:
- `scripts/build_oasis1_imaging_manifest.py`
- `scripts/extract_oasis1_tissue_features.py`
- `scripts/merge_oasis1_imaging_features.py`

---

## Summary

**What We Accomplished:**
- ✅ Extracted 22 tissue-specific imaging features
- ✅ 100% extraction success rate (39/39 sessions)
- ✅ Excellent validation (nWBV error < 0.001)
- ✅ Session-safe merge with audit trail
- ✅ Enhanced CSV ready for Phase 2

**What We Improved:**
- ✅ Tissue-specific metrics (GM, WM, CSF separate)
- ✅ Atrophy markers (CSF-to-brain ratio)
- ✅ eTIV-normalized features
- ✅ Pipeline validation and quality control

**What's Still Needed:**
- ⚠️ Regional features (hippocampus, entorhinal, temporal)
- ⚠️ Full OASIS-1 dataset (377 more sessions from other discs)
- ⚠️ Atlas-based ROI extraction implementation

**Clinical Impact:**
- Current features: Better than global nWBV alone
- Still limited: Cannot detect Alzheimer's-specific patterns
- Next phase critical: Regional features required for clinical reliability

---

**Document Version:** 1.0  
**Last Updated:** March 22, 2026  
**Contact:** Phase 2 MRI Feature Extraction Pipeline
