"""
Tissue feature extraction from OASIS-1 FSL segmentation data.

Extracts GM/WM/CSF tissue composition features from:
1. FSL_SEG .txt files (volume statistics)
2. FSL_SEG segmentation images (voxel-level analysis)

Features extracted:
- Tissue voxel counts and volumes
- Tissue fractions
- Brain parenchyma metrics
- Reconstructed nWBV for validation
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
import re
import numpy as np
import nibabel as nib
import pandas as pd

logger = logging.getLogger(__name__)


def parse_fsl_seg_txt(txt_path: Path) -> dict:
    """
    Parse FSL FAST segmentation .txt file to extract tissue volumes.
    
    The .txt file contains a line like:
    "Volumes:	428136.9  	740840.2  	498315.1  	0.743214"
    Where values are: CSF_volume, GM_volume, WM_volume, brain_percentage
    
    Args:
        txt_path: Path to FSL_SEG .txt file
        
    Returns:
        Dictionary with parsed tissue volumes
    """
    txt_path = Path(txt_path)
    
    if not txt_path.exists():
        return {
            'status': 'missing',
            'error': f'File not found: {txt_path}',
            'csf_vol_mm3': None,
            'gm_vol_mm3': None,
            'wm_vol_mm3': None,
            'brain_percentage': None
        }
    
    try:
        with open(txt_path, 'r') as f:
            content = f.read()
        
        # Find the "Volumes:" line
        volume_pattern = r'Volumes:\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
        match = re.search(volume_pattern, content)
        
        if match:
            csf_vol = float(match.group(1))
            gm_vol = float(match.group(2))
            wm_vol = float(match.group(3))
            brain_pct = float(match.group(4))
            
            return {
                'status': 'success',
                'error': None,
                'csf_vol_mm3': csf_vol,
                'gm_vol_mm3': gm_vol,
                'wm_vol_mm3': wm_vol,
                'brain_percentage': brain_pct
            }
        else:
            return {
                'status': 'parse_failed',
                'error': 'Could not find Volumes line in .txt file',
                'csf_vol_mm3': None,
                'gm_vol_mm3': None,
                'wm_vol_mm3': None,
                'brain_percentage': None
            }
            
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'csf_vol_mm3': None,
            'gm_vol_mm3': None,
            'wm_vol_mm3': None,
            'brain_percentage': None
        }


def extract_tissue_voxel_counts(seg_image_path: Path) -> dict:
    """
    Extract tissue voxel counts from FSL segmentation image.
    
    FSL FAST produces a 3-class segmentation where:
    - 0 = CSF
    - 1 = GM (gray matter)
    - 2 = WM (white matter)
    
    Args:
        seg_image_path: Path to segmentation image (.hdr or .nii.gz)
        
    Returns:
        Dictionary with voxel counts and derived metrics
    """
    seg_image_path = Path(seg_image_path)
    
    if not seg_image_path.exists():
        return {
            'status': 'missing',
            'error': f'File not found: {seg_image_path}',
            'csf_voxels': None,
            'gm_voxels': None,
            'wm_voxels': None
        }
    
    try:
        img = nib.load(str(seg_image_path))
        data = img.get_fdata()
        
        # Get voxel dimensions for volume calculation
        voxel_dims = img.header.get_zooms()[:3]
        voxel_vol_mm3 = np.prod(voxel_dims)
        
        # Count voxels for each tissue class
        csf_voxels = int(np.sum(data == 0))
        gm_voxels = int(np.sum(data == 1))
        wm_voxels = int(np.sum(data == 2))
        
        return {
            'status': 'success',
            'error': None,
            'csf_voxels': csf_voxels,
            'gm_voxels': gm_voxels,
            'wm_voxels': wm_voxels,
            'voxel_vol_mm3': float(voxel_vol_mm3)
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'csf_voxels': None,
            'gm_voxels': None,
            'wm_voxels': None,
            'voxel_vol_mm3': None
        }


def compute_tissue_features(
    txt_path: Optional[Path],
    seg_image_path: Optional[Path],
    csv_nwbv: Optional[float] = None,
    csv_etiv: Optional[float] = None
) -> dict:
    """
    Compute comprehensive tissue features from FSL segmentation data.
    
    Args:
        txt_path: Path to FSL_SEG .txt file
        seg_image_path: Path to segmentation image
        csv_nwbv: nWBV value from CSV (for validation)
        csv_etiv: eTIV value from CSV (for normalization)
        
    Returns:
        Dictionary with all tissue features
    """
    features = {}
    
    # Parse .txt file for volumes
    txt_results = parse_fsl_seg_txt(txt_path) if txt_path else {'status': 'not_provided'}
    
    # Extract voxel counts from segmentation image
    seg_results = extract_tissue_voxel_counts(seg_image_path) if seg_image_path else {'status': 'not_provided'}
    
    # Store raw values
    features['txt_status'] = txt_results.get('status')
    features['seg_status'] = seg_results.get('status')
    
    # Get volumes from .txt file (preferred source)
    csf_vol = txt_results.get('csf_vol_mm3')
    gm_vol = txt_results.get('gm_vol_mm3')
    wm_vol = txt_results.get('wm_vol_mm3')
    
    # If .txt parsing failed, compute from voxel counts
    if csf_vol is None and seg_results.get('status') == 'success':
        voxel_vol = seg_results.get('voxel_vol_mm3', 1.0)
        csf_vol = seg_results.get('csf_voxels', 0) * voxel_vol
        gm_vol = seg_results.get('gm_voxels', 0) * voxel_vol
        wm_vol = seg_results.get('wm_voxels', 0) * voxel_vol
        features['volume_source'] = 'voxel_counts'
    else:
        features['volume_source'] = 'txt_file'
    
    # Store voxel counts
    features['csf_voxels'] = seg_results.get('csf_voxels')
    features['gm_voxels'] = seg_results.get('gm_voxels')
    features['wm_voxels'] = seg_results.get('wm_voxels')
    
    # Store volumes
    features['csf_vol_mm3'] = csf_vol
    features['gm_vol_mm3'] = gm_vol
    features['wm_vol_mm3'] = wm_vol
    
    # Compute derived features if we have valid volumes
    if csf_vol is not None and gm_vol is not None and wm_vol is not None:
        brain_parenchyma_vol = gm_vol + wm_vol
        total_segmented_vol = csf_vol + gm_vol + wm_vol
        
        features['brain_parenchyma_vol_mm3'] = brain_parenchyma_vol
        features['total_segmented_vol_mm3'] = total_segmented_vol
        
        # Compute fractions
        if total_segmented_vol > 0:
            features['csf_frac'] = csf_vol / total_segmented_vol
            features['gm_frac'] = gm_vol / total_segmented_vol
            features['wm_frac'] = wm_vol / total_segmented_vol
            features['brain_parenchyma_frac'] = brain_parenchyma_vol / total_segmented_vol
        else:
            features['csf_frac'] = None
            features['gm_frac'] = None
            features['wm_frac'] = None
            features['brain_parenchyma_frac'] = None
        
        # Compute ratios
        if brain_parenchyma_vol > 0:
            features['csf_to_brain_ratio'] = csf_vol / brain_parenchyma_vol
        else:
            features['csf_to_brain_ratio'] = None
        
        if wm_vol > 0:
            features['gm_wm_ratio'] = gm_vol / wm_vol
        else:
            features['gm_wm_ratio'] = None
        
        # Compute reconstructed nWBV
        if total_segmented_vol > 0:
            features['reconstructed_nwbv'] = brain_parenchyma_vol / total_segmented_vol
        else:
            features['reconstructed_nwbv'] = None
        
        # Validate against CSV nWBV if provided
        if csv_nwbv is not None and features['reconstructed_nwbv'] is not None:
            features['csv_nwbv'] = csv_nwbv
            features['reconstructed_nwbv_abs_error'] = abs(features['reconstructed_nwbv'] - csv_nwbv)
            features['reconstructed_nwbv_pct_error'] = (features['reconstructed_nwbv_abs_error'] / csv_nwbv) * 100
        else:
            features['csv_nwbv'] = csv_nwbv
            features['reconstructed_nwbv_abs_error'] = None
            features['reconstructed_nwbv_pct_error'] = None
        
        # Normalize by eTIV if provided
        if csv_etiv is not None and csv_etiv > 0:
            features['brain_parenchyma_to_etiv'] = brain_parenchyma_vol / csv_etiv
            features['gm_to_etiv'] = gm_vol / csv_etiv
            features['wm_to_etiv'] = wm_vol / csv_etiv
            features['csf_to_etiv'] = csf_vol / csv_etiv
        else:
            features['brain_parenchyma_to_etiv'] = None
            features['gm_to_etiv'] = None
            features['wm_to_etiv'] = None
            features['csf_to_etiv'] = None
    else:
        # Set all derived features to None if volumes are missing
        features['brain_parenchyma_vol_mm3'] = None
        features['total_segmented_vol_mm3'] = None
        features['csf_frac'] = None
        features['gm_frac'] = None
        features['wm_frac'] = None
        features['brain_parenchyma_frac'] = None
        features['csf_to_brain_ratio'] = None
        features['gm_wm_ratio'] = None
        features['reconstructed_nwbv'] = None
        features['csv_nwbv'] = csv_nwbv
        features['reconstructed_nwbv_abs_error'] = None
        features['reconstructed_nwbv_pct_error'] = None
        features['brain_parenchyma_to_etiv'] = None
        features['gm_to_etiv'] = None
        features['wm_to_etiv'] = None
        features['csf_to_etiv'] = None
    
    return features


def extract_session_tissue_features(
    session_id: str,
    fsl_seg_txt_path: Optional[Path],
    fsl_seg_image_path: Optional[Path],
    csv_nwbv: Optional[float] = None,
    csv_etiv: Optional[float] = None
) -> dict:
    """
    Extract all tissue features for a single session.
    
    Args:
        session_id: Session identifier (e.g., 'OAS1_0001_MR1')
        fsl_seg_txt_path: Path to FSL_SEG .txt file
        fsl_seg_image_path: Path to FSL_SEG segmentation image
        csv_nwbv: nWBV from CSV for validation
        csv_etiv: eTIV from CSV for normalization
        
    Returns:
        Dictionary with session_id and all tissue features
    """
    features = compute_tissue_features(
        fsl_seg_txt_path,
        fsl_seg_image_path,
        csv_nwbv,
        csv_etiv
    )
    
    features['session_id'] = session_id
    features['canonical_session_key'] = session_id
    
    return features


def validate_tissue_features(features_df: pd.DataFrame) -> dict:
    """
    Validate extracted tissue features.
    
    Args:
        features_df: DataFrame with tissue features
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        'total_sessions': len(features_df),
        'successful_extractions': 0,
        'failed_extractions': 0,
        'nwbv_validation': {}
    }
    
    # Check extraction success
    if 'txt_status' in features_df.columns:
        validation['successful_extractions'] = (features_df['txt_status'] == 'success').sum()
        validation['failed_extractions'] = (features_df['txt_status'] != 'success').sum()
    
    # Validate tissue fractions sum to ~1
    if all(col in features_df.columns for col in ['csf_frac', 'gm_frac', 'wm_frac']):
        frac_sum = features_df[['csf_frac', 'gm_frac', 'wm_frac']].sum(axis=1)
        validation['fraction_sum_mean'] = float(frac_sum.mean())
        validation['fraction_sum_std'] = float(frac_sum.std())
        validation['fraction_sum_ok'] = ((frac_sum > 0.99) & (frac_sum < 1.01)).sum()
    
    # Validate reconstructed nWBV
    if 'reconstructed_nwbv_abs_error' in features_df.columns:
        valid_errors = features_df['reconstructed_nwbv_abs_error'].dropna()
        if len(valid_errors) > 0:
            validation['nwbv_validation'] = {
                'n_validated': len(valid_errors),
                'mean_abs_error': float(valid_errors.mean()),
                'median_abs_error': float(valid_errors.median()),
                'max_abs_error': float(valid_errors.max()),
                'error_lt_0.01': (valid_errors < 0.01).sum(),
                'error_lt_0.05': (valid_errors < 0.05).sum()
            }
    
    return validation
