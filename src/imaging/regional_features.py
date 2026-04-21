"""
Improved regional feature extraction using tissue segmentation within ROI masks.

This version extracts tissue-specific volumes:
- Hippocampus: GM (gray matter) within hippocampal ROI
- Ventricles: CSF within ventricular ROI  
- Temporal cortex: GM within temporal ROI

This provides session-specific volumes with proper variance.
"""

import logging
from pathlib import Path
from typing import Optional, Dict
import pandas as pd
import numpy as np

from .atlas_utils import extract_bilateral_roi_volumes_from_segmentation

logger = logging.getLogger(__name__)


def extract_session_regional_features_v2(
    session_id: str,
    fsl_seg_image_path: Optional[Path],
    csv_etiv: Optional[float] = None
) -> Dict[str, any]:
    """
    Extract regional features using tissue segmentation within ROI masks.
    
    Args:
        session_id: Session identifier (e.g., 'OAS1_0001_MR1')
        fsl_seg_image_path: Path to FSL_SEG segmentation image (.hdr or .nii.gz)
        csv_etiv: eTIV from CSV for normalization
        
    Returns:
        Dictionary with session_id and all regional features
    """
    features = {
        'session_id': session_id,
        'canonical_session_key': session_id,
        'regional_extraction_status': 'pending'
    }
    
    # Check if segmentation image exists
    if fsl_seg_image_path is None or not Path(fsl_seg_image_path).exists():
        features['regional_extraction_status'] = 'missing_image'
        features['regional_extraction_error'] = f'FSL_SEG image not found'
        return _add_null_regional_features(features)
    
    try:
        # Extract hippocampal volumes (GM only - label 2 in FSL FAST)
        logger.info(f"Extracting hippocampus (GM) for {session_id}")
        hippo_results = extract_bilateral_roi_volumes_from_segmentation(
            fsl_seg_image_path,
            'hippocampus',
            target_tissue_label=2  # Gray matter (FSL FAST: 1=CSF, 2=GM, 3=WM)
        )
        features.update(hippo_results)
        
        # Extract ventricular volumes (CSF only - label 1 in FSL FAST)
        logger.info(f"Extracting ventricles (CSF) for {session_id}")
        ventricle_results = extract_bilateral_roi_volumes_from_segmentation(
            fsl_seg_image_path,
            'ventricle',
            target_tissue_label=1  # CSF (FSL FAST: 1=CSF, 2=GM, 3=WM)
        )
        features.update(ventricle_results)
        
        # Extract entorhinal cortex (GM only - label 2 in FSL FAST)
        logger.info(f"Extracting entorhinal cortex (GM) for {session_id}")
        entorhinal_results = extract_bilateral_roi_volumes_from_segmentation(
            fsl_seg_image_path,
            'entorhinal',
            target_tissue_label=2  # Gray matter (FSL FAST: 1=CSF, 2=GM, 3=WM)
        )
        features.update(entorhinal_results)
        
        # Extract inferior temporal gyrus (GM only - label 2 in FSL FAST)
        logger.info(f"Extracting inferior temporal (GM) for {session_id}")
        inf_temporal_results = extract_bilateral_roi_volumes_from_segmentation(
            fsl_seg_image_path,
            'inferior_temporal',
            target_tissue_label=2  # Gray matter (FSL FAST: 1=CSF, 2=GM, 3=WM)
        )
        features.update(inf_temporal_results)
        
        # Extract middle temporal gyrus (GM only - label 2 in FSL FAST)
        logger.info(f"Extracting middle temporal (GM) for {session_id}")
        mid_temporal_results = extract_bilateral_roi_volumes_from_segmentation(
            fsl_seg_image_path,
            'middle_temporal',
            target_tissue_label=2  # Gray matter (FSL FAST: 1=CSF, 2=GM, 3=WM)
        )
        features.update(mid_temporal_results)
        
        # Normalize by eTIV if available
        if csv_etiv is not None and csv_etiv > 0:
            features = _add_etiv_normalized_regional_features(features, csv_etiv)
        
        # Calculate composite temporal lobe metrics
        features = _calculate_composite_temporal_metrics(features)
        
        # Overall status
        if hippo_results.get('status') == 'success' and ventricle_results.get('status') == 'success':
            features['regional_extraction_status'] = 'success'
        elif hippo_results.get('status') in ['success', 'partial'] or ventricle_results.get('status') in ['success', 'partial']:
            features['regional_extraction_status'] = 'partial'
        else:
            features['regional_extraction_status'] = 'failed'
        
        # Clean up intermediate status fields
        for key in list(features.keys()):
            if key == 'status' or key == 'error':
                del features[key]
        
    except Exception as e:
        logger.error(f"Regional extraction failed for {session_id}: {str(e)}")
        features['regional_extraction_status'] = 'error'
        features['regional_extraction_error'] = str(e)
        features = _add_null_regional_features(features)
    
    return features


def _add_null_regional_features(features: Dict) -> Dict:
    """Add null values for all regional features."""
    
    # Hippocampus
    features['hippocampus_left_volume_mm3'] = None
    features['hippocampus_right_volume_mm3'] = None
    features['hippocampus_bilateral_volume_mm3'] = None
    features['hippocampus_asymmetry_index'] = None
    features['hippocampus_laterality_ratio'] = None
    
    # Ventricles
    features['ventricle_left_volume_mm3'] = None
    features['ventricle_right_volume_mm3'] = None
    features['ventricle_bilateral_volume_mm3'] = None
    features['ventricle_asymmetry_index'] = None
    features['ventricle_laterality_ratio'] = None
    
    # Entorhinal
    features['entorhinal_left_volume_mm3'] = None
    features['entorhinal_right_volume_mm3'] = None
    features['entorhinal_bilateral_volume_mm3'] = None
    features['entorhinal_asymmetry_index'] = None
    
    # Temporal
    features['inferior_temporal_left_volume_mm3'] = None
    features['inferior_temporal_right_volume_mm3'] = None
    features['inferior_temporal_bilateral_volume_mm3'] = None
    features['middle_temporal_left_volume_mm3'] = None
    features['middle_temporal_right_volume_mm3'] = None
    features['middle_temporal_bilateral_volume_mm3'] = None
    
    # Composite
    features['total_temporal_lobe_volume_mm3'] = None
    features['hippocampus_to_temporal_ratio'] = None
    
    return features


def _add_etiv_normalized_regional_features(features: Dict, etiv: float) -> Dict:
    """Add eTIV-normalized versions of regional features."""
    
    # Hippocampus
    if features.get('hippocampus_bilateral_volume_mm3') is not None:
        features['hippocampus_bilateral_to_etiv'] = features['hippocampus_bilateral_volume_mm3'] / etiv
        features['hippocampus_left_to_etiv'] = features['hippocampus_left_volume_mm3'] / etiv
        features['hippocampus_right_to_etiv'] = features['hippocampus_right_volume_mm3'] / etiv
    
    # Ventricles
    if features.get('ventricle_bilateral_volume_mm3') is not None:
        features['ventricle_bilateral_to_etiv'] = features['ventricle_bilateral_volume_mm3'] / etiv
        features['ventricle_left_to_etiv'] = features['ventricle_left_volume_mm3'] / etiv
        features['ventricle_right_to_etiv'] = features['ventricle_right_volume_mm3'] / etiv
    
    # Entorhinal
    if features.get('entorhinal_bilateral_volume_mm3') is not None:
        features['entorhinal_bilateral_to_etiv'] = features['entorhinal_bilateral_volume_mm3'] / etiv
    
    # Temporal
    if features.get('total_temporal_lobe_volume_mm3') is not None:
        features['total_temporal_to_etiv'] = features['total_temporal_lobe_volume_mm3'] / etiv
    
    return features


def _calculate_composite_temporal_metrics(features: Dict) -> Dict:
    """Calculate composite metrics for temporal lobe regions."""
    
    # Total temporal lobe volume (sum of all temporal ROIs)
    temporal_components = [
        features.get('entorhinal_bilateral_volume_mm3'),
        features.get('inferior_temporal_bilateral_volume_mm3'),
        features.get('middle_temporal_bilateral_volume_mm3')
    ]
    
    # Filter out None values
    valid_components = [v for v in temporal_components if v is not None]
    
    if len(valid_components) > 0:
        features['total_temporal_lobe_volume_mm3'] = sum(valid_components)
    else:
        features['total_temporal_lobe_volume_mm3'] = None
    
    # Hippocampus to temporal lobe ratio
    hippo_vol = features.get('hippocampus_bilateral_volume_mm3')
    temporal_vol = features.get('total_temporal_lobe_volume_mm3')
    
    if hippo_vol is not None and temporal_vol is not None and temporal_vol > 0:
        features['hippocampus_to_temporal_ratio'] = hippo_vol / temporal_vol
    else:
        features['hippocampus_to_temporal_ratio'] = None
    
    return features


def validate_regional_features(features_df: pd.DataFrame) -> Dict:
    """
    Validate extracted regional features.
    
    Args:
        features_df: DataFrame with regional features
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        'total_sessions': len(features_df),
        'successful_extractions': 0,
        'partial_extractions': 0,
        'failed_extractions': 0
    }
    
    # Check extraction success
    if 'regional_extraction_status' in features_df.columns:
        validation['successful_extractions'] = (features_df['regional_extraction_status'] == 'success').sum()
        validation['partial_extractions'] = (features_df['regional_extraction_status'] == 'partial').sum()
        validation['failed_extractions'] = (features_df['regional_extraction_status'] == 'failed').sum()
    
    # Check hippocampus extraction
    if 'hippocampus_bilateral_volume_mm3' in features_df.columns:
        hippo_valid = features_df['hippocampus_bilateral_volume_mm3'].notna()
        validation['hippocampus_extracted'] = int(hippo_valid.sum())
        
        if hippo_valid.sum() > 0:
            hippo_vols = features_df.loc[hippo_valid, 'hippocampus_bilateral_volume_mm3']
            validation['hippocampus_volume_stats'] = {
                'mean': float(hippo_vols.mean()),
                'std': float(hippo_vols.std()),
                'min': float(hippo_vols.min()),
                'max': float(hippo_vols.max()),
                'cv': float(hippo_vols.std() / hippo_vols.mean()) if hippo_vols.mean() > 0 else 0.0
            }
    
    # Check ventricle extraction
    if 'ventricle_bilateral_volume_mm3' in features_df.columns:
        vent_valid = features_df['ventricle_bilateral_volume_mm3'].notna()
        validation['ventricle_extracted'] = int(vent_valid.sum())
        
        if vent_valid.sum() > 0:
            vent_vols = features_df.loc[vent_valid, 'ventricle_bilateral_volume_mm3']
            validation['ventricle_volume_stats'] = {
                'mean': float(vent_vols.mean()),
                'std': float(vent_vols.std()),
                'min': float(vent_vols.min()),
                'max': float(vent_vols.max()),
                'cv': float(vent_vols.std() / vent_vols.mean()) if vent_vols.mean() > 0 else 0.0
            }
    
    # Check asymmetry indices
    if 'hippocampus_asymmetry_index' in features_df.columns:
        asym_valid = features_df['hippocampus_asymmetry_index'].notna()
        if asym_valid.sum() > 0:
            asym_vals = features_df.loc[asym_valid, 'hippocampus_asymmetry_index']
            validation['hippocampus_asymmetry_stats'] = {
                'mean': float(asym_vals.mean()),
                'std': float(asym_vals.std()),
                'abs_mean': float(asym_vals.abs().mean())
            }
    
    return validation
