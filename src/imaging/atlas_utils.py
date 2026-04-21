"""
Improved atlas utilities using tissue segmentation within ROI masks.

This version extracts tissue-specific volumes within anatomical ROIs:
- Hippocampus: Gray matter within hippocampal ROI
- Ventricles: CSF within ventricular ROI
- Temporal cortex: Gray matter within temporal ROI

This provides session-specific volumes that vary based on actual tissue composition.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, Dict
import numpy as np
import nibabel as nib

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Talairach coordinate definitions for AD-relevant ROIs
# Coordinates are approximate bounding boxes in MNI/Talairach space (mm).
# Each ROI has left/right hemispheric definitions.
# ---------------------------------------------------------------------------

TALAIRACH_ROI_COORDS = {
    # Hippocampus — medial temporal lobe, critical for memory
    'hippocampus_left':  {'x_range': (-35, -20), 'y_range': (-40, -10), 'z_range': (-20, -5)},
    'hippocampus_right': {'x_range': (20, 35),   'y_range': (-40, -10), 'z_range': (-20, -5)},

    # Lateral ventricles — CSF-filled cavities, expand with atrophy
    'ventricle_left':  {'x_range': (-35, -2),  'y_range': (-60, 20),  'z_range': (-5, 30)},
    'ventricle_right': {'x_range': (2, 35),    'y_range': (-60, 20),  'z_range': (-5, 30)},

    # Entorhinal cortex — earliest cortical region affected in AD
    'entorhinal_left':  {'x_range': (-30, -15), 'y_range': (-15, 5),  'z_range': (-35, -15)},
    'entorhinal_right': {'x_range': (15, 30),   'y_range': (-15, 5),  'z_range': (-35, -15)},

    # Inferior temporal gyrus — involved in visual processing and semantic memory
    'inferior_temporal_left':  {'x_range': (-55, -30), 'y_range': (-60, -5), 'z_range': (-35, -10)},
    'inferior_temporal_right': {'x_range': (30, 55),   'y_range': (-60, -5), 'z_range': (-35, -10)},

    # Middle temporal gyrus — language and semantic processing
    'middle_temporal_left':  {'x_range': (-65, -40), 'y_range': (-60, 0),  'z_range': (-15, 10)},
    'middle_temporal_right': {'x_range': (40, 65),   'y_range': (-60, 0),  'z_range': (-15, 10)},
}


def talairach_coords_to_voxel_indices(
    coords: Tuple[float, float, float],
    affine: np.ndarray,
    image_shape: Tuple[int, int, int]
) -> Tuple[int, int, int]:
    """
    Convert Talairach/MNI mm coordinates to voxel indices using the image affine.

    Args:
        coords: (x, y, z) in mm
        affine: 4×4 affine matrix from the NIfTI header
        image_shape: (i, j, k) dimensions of the image

    Returns:
        (i, j, k) integer voxel indices, clamped to valid range
    """
    inv_affine = np.linalg.inv(affine)
    mm_coord = np.array([coords[0], coords[1], coords[2], 1.0])
    voxel_coord = inv_affine.dot(mm_coord)[:3]

    # Clamp to image bounds
    i = int(np.clip(np.round(voxel_coord[0]), 0, image_shape[0] - 1))
    j = int(np.clip(np.round(voxel_coord[1]), 0, image_shape[1] - 1))
    k = int(np.clip(np.round(voxel_coord[2]), 0, image_shape[2] - 1))

    return (i, j, k)


def create_roi_mask_from_talairach_coords(
    roi_name: str,
    image_shape: Tuple[int, int, int],
    affine: np.ndarray
) -> Tuple[Optional[np.ndarray], str]:
    """
    Create a binary ROI mask based on Talairach coordinate ranges.
    
    Args:
        roi_name: Name of ROI (must be in TALAIRACH_ROI_COORDS)
        image_shape: Target image dimensions
        affine: Image affine transformation matrix
        
    Returns:
        Tuple of (mask array or None, status message)
    """
    if roi_name not in TALAIRACH_ROI_COORDS:
        return None, f"ROI '{roi_name}' not found in atlas definitions"
    
    roi_def = TALAIRACH_ROI_COORDS[roi_name]
    x_range = roi_def['x_range']
    y_range = roi_def['y_range']
    z_range = roi_def['z_range']
    
    try:
        # Create empty mask
        mask = np.zeros(image_shape, dtype=np.uint8)
        
        # Convert coordinate ranges to voxel indices
        corner_min = talairach_coords_to_voxel_indices(
            (x_range[0], y_range[0], z_range[0]),
            affine,
            image_shape
        )
        corner_max = talairach_coords_to_voxel_indices(
            (x_range[1], y_range[1], z_range[1]),
            affine,
            image_shape
        )
        
        # Ensure min < max for each dimension
        i_min, i_max = min(corner_min[0], corner_max[0]), max(corner_min[0], corner_max[0])
        j_min, j_max = min(corner_min[1], corner_max[1]), max(corner_min[1], corner_max[1])
        k_min, k_max = min(corner_min[2], corner_max[2]), max(corner_min[2], corner_max[2])
        
        # Fill the mask
        mask[i_min:i_max+1, j_min:j_max+1, k_min:k_max+1] = 1
        
        voxel_count = np.sum(mask)
        
        return mask, f"ROI mask created: {voxel_count} voxels"
        
    except Exception as e:
        return None, f"Failed to create ROI mask: {str(e)}"


def extract_tissue_specific_roi_volume(
    seg_image_data: np.ndarray,
    roi_mask: np.ndarray,
    voxel_dims: Tuple[float, float, float],
    target_tissue_label: int
) -> Dict[str, float]:
    """
    Extract volume of specific tissue type within an ROI.
    
    Args:
        seg_image_data: 3D segmentation array (FSL FAST: 1=CSF, 2=GM, 3=WM, 0=background)
        roi_mask: Binary ROI mask
        voxel_dims: Voxel dimensions (mm)
        target_tissue_label: Tissue label to extract (1=CSF, 2=GM, 3=WM)
        
    Returns:
        Dictionary with volume statistics
    """
    if seg_image_data.shape != roi_mask.shape:
        raise ValueError(f"Segmentation shape {seg_image_data.shape} != mask shape {roi_mask.shape}")
    
    # Calculate voxel volume
    voxel_vol_mm3 = np.prod(voxel_dims)
    
    # Extract voxels within ROI
    roi_voxels = seg_image_data[roi_mask > 0]
    
    # Count voxels of target tissue type
    tissue_voxels = np.sum(roi_voxels == target_tissue_label)
    
    # Calculate volume
    roi_volume_mm3 = tissue_voxels * voxel_vol_mm3
    
    # Calculate tissue fractions within ROI
    total_roi_voxels = len(roi_voxels)
    if total_roi_voxels > 0:
        tissue_fraction = tissue_voxels / total_roi_voxels
    else:
        tissue_fraction = 0.0
    
    # Count all tissue types within ROI for reference (FSL FAST labels: 1=CSF, 2=GM, 3=WM)
    csf_voxels = np.sum(roi_voxels == 1)
    gm_voxels = np.sum(roi_voxels == 2)
    wm_voxels = np.sum(roi_voxels == 3)
    background_voxels = np.sum(roi_voxels == 0)
    
    return {
        'roi_tissue_voxel_count': int(tissue_voxels),
        'roi_tissue_volume_mm3': float(roi_volume_mm3),
        'roi_tissue_fraction': float(tissue_fraction),
        'roi_total_voxels': int(total_roi_voxels),
        'roi_csf_voxels': int(csf_voxels),
        'roi_gm_voxels': int(gm_voxels),
        'roi_wm_voxels': int(wm_voxels),
        'roi_background_voxels': int(background_voxels),
        'target_tissue_label': target_tissue_label
    }


def extract_bilateral_roi_volumes_from_segmentation(
    seg_image_path: Path,
    roi_base_name: str,
    target_tissue_label: int
) -> Dict[str, any]:
    """
    Extract bilateral ROI volumes using tissue segmentation.
    
    Args:
        seg_image_path: Path to FSL_SEG segmentation image
        roi_base_name: Base name of ROI (e.g., 'hippocampus', 'ventricle')
        target_tissue_label: Tissue to extract (0=CSF, 1=GM, 2=WM)
        
    Returns:
        Dictionary with left, right, and combined statistics
    """
    results = {
        'status': 'pending',
        'error': None
    }
    
    # Load segmentation image
    try:
        img = nib.load(str(seg_image_path))
        seg_data = img.get_fdata()
        affine = img.affine
        voxel_dims = img.header.get_zooms()[:3]
        
    except Exception as e:
        results['status'] = 'failed'
        results['error'] = f"Failed to load segmentation: {str(e)}"
        return results
    
    # Extract left ROI
    left_roi_name = f"{roi_base_name}_left"
    left_mask, left_msg = create_roi_mask_from_talairach_coords(
        left_roi_name,
        seg_data.shape,
        affine
    )
    
    if left_mask is not None:
        left_stats = extract_tissue_specific_roi_volume(
            seg_data, left_mask, voxel_dims, target_tissue_label
        )
        results[f'{roi_base_name}_left_volume_mm3'] = left_stats['roi_tissue_volume_mm3']
        results[f'{roi_base_name}_left_voxels'] = left_stats['roi_tissue_voxel_count']
        results[f'{roi_base_name}_left_tissue_fraction'] = left_stats['roi_tissue_fraction']
        results[f'{roi_base_name}_left_gm_voxels'] = left_stats['roi_gm_voxels']
        results[f'{roi_base_name}_left_wm_voxels'] = left_stats['roi_wm_voxels']
        results[f'{roi_base_name}_left_csf_voxels'] = left_stats['roi_csf_voxels']
    else:
        results[f'{roi_base_name}_left_volume_mm3'] = None
        results[f'{roi_base_name}_left_voxels'] = None
        results[f'{roi_base_name}_left_tissue_fraction'] = None
        results['error'] = left_msg
    
    # Extract right ROI
    right_roi_name = f"{roi_base_name}_right"
    right_mask, right_msg = create_roi_mask_from_talairach_coords(
        right_roi_name,
        seg_data.shape,
        affine
    )
    
    if right_mask is not None:
        right_stats = extract_tissue_specific_roi_volume(
            seg_data, right_mask, voxel_dims, target_tissue_label
        )
        results[f'{roi_base_name}_right_volume_mm3'] = right_stats['roi_tissue_volume_mm3']
        results[f'{roi_base_name}_right_voxels'] = right_stats['roi_tissue_voxel_count']
        results[f'{roi_base_name}_right_tissue_fraction'] = right_stats['roi_tissue_fraction']
        results[f'{roi_base_name}_right_gm_voxels'] = right_stats['roi_gm_voxels']
        results[f'{roi_base_name}_right_wm_voxels'] = right_stats['roi_wm_voxels']
        results[f'{roi_base_name}_right_csf_voxels'] = right_stats['roi_csf_voxels']
    else:
        results[f'{roi_base_name}_right_volume_mm3'] = None
        results[f'{roi_base_name}_right_voxels'] = None
        results[f'{roi_base_name}_right_tissue_fraction'] = None
        if results['error'] is None:
            results['error'] = right_msg
    
    # Calculate combined and asymmetry metrics
    if (left_mask is not None and right_mask is not None):
        left_vol = results[f'{roi_base_name}_left_volume_mm3']
        right_vol = results[f'{roi_base_name}_right_volume_mm3']
        
        # Total bilateral volume
        results[f'{roi_base_name}_bilateral_volume_mm3'] = left_vol + right_vol
        
        # Asymmetry index: (L - R) / (L + R)
        if (left_vol + right_vol) > 0:
            results[f'{roi_base_name}_asymmetry_index'] = (left_vol - right_vol) / (left_vol + right_vol)
        else:
            results[f'{roi_base_name}_asymmetry_index'] = 0.0
        
        # Laterality ratio: L / R
        if right_vol > 0:
            results[f'{roi_base_name}_laterality_ratio'] = left_vol / right_vol
        else:
            results[f'{roi_base_name}_laterality_ratio'] = None
        
        results['status'] = 'success'
    else:
        results['status'] = 'partial' if (left_mask is not None or right_mask is not None) else 'failed'
    
    return results
