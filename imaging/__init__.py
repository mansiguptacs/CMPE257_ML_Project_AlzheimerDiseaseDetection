"""
OASIS-1 MRI Feature Extraction Module

This module provides utilities for extracting clinically interpretable
structural imaging biomarkers from OASIS-1 MRI data.

Modules:
- io_utils: Analyze-to-NIfTI conversion utilities
- qc: Quality control visualization and statistics
- tissue_features: GM/WM/CSF tissue composition extraction
- regional_features: Atlas-based ROI extraction (hippocampus, ventricles)
- atlas_utils: Atlas loading and ROI mapping utilities
- merge_utils: Session-safe merge validation
"""

__version__ = '0.1.0'