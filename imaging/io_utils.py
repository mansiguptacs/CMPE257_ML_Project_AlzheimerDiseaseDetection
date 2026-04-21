"""
I/O utilities for OASIS-1 MRI data processing.

Provides functions for:
- Analyze format (.hdr/.img) to NIfTI (.nii.gz) conversion
- File discovery and validation
- Image loading with error handling
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


def convert_analyze_to_nifti(
    hdr_path: Path,
    output_path: Optional[Path] = None,
    overwrite: bool = False
) -> Tuple[bool, str]:
    """
    Convert Analyze format (.hdr/.img) to NIfTI (.nii.gz).
    
    Args:
        hdr_path: Path to .hdr file
        output_path: Output .nii.gz path (default: same location with .nii.gz extension)
        overwrite: Whether to overwrite existing output file
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    hdr_path = Path(hdr_path)
    
    if not hdr_path.exists():
        return False, f"Header file not found: {hdr_path}"
    
    img_path = hdr_path.with_suffix('.img')
    if not img_path.exists():
        return False, f"Image file not found: {img_path}"
    
    if output_path is None:
        output_path = hdr_path.with_suffix('.nii.gz')
    else:
        output_path = Path(output_path)
    
    if output_path.exists() and not overwrite:
        return True, f"Output already exists (skipped): {output_path}"
    
    try:
        # Load Analyze format
        img = nib.load(str(hdr_path))
        
        # Convert to NIfTI format
        nifti_img = nib.Nifti1Image(
            img.get_fdata(),
            img.affine,
            img.header
        )
        
        # Save as compressed NIfTI
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nifti_img, str(output_path))
        
        return True, f"Converted successfully: {output_path}"
        
    except Exception as e:
        return False, f"Conversion failed: {str(e)}"


def load_image_safe(
    image_path: Path,
    return_data: bool = False
) -> Tuple[Optional[nib.Nifti1Image], Optional[np.ndarray], str]:
    """
    Safely load a NIfTI or Analyze image with error handling.
    
    Args:
        image_path: Path to image file (.nii.gz, .nii, .hdr)
        return_data: Whether to also return the data array
        
    Returns:
        Tuple of (image: nibabel image or None, data: numpy array or None, message: str)
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        return None, None, f"File not found: {image_path}"
    
    try:
        img = nib.load(str(image_path))
        
        if return_data:
            data = img.get_fdata()
            return img, data, "Loaded successfully"
        else:
            return img, None, "Loaded successfully"
            
    except Exception as e:
        return None, None, f"Load failed: {str(e)}"


def get_image_info(image_path: Path) -> dict:
    """
    Extract basic information from an image file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary with image metadata
    """
    img, data, msg = load_image_safe(image_path, return_data=True)
    
    if img is None:
        return {
            'path': str(image_path),
            'status': 'failed',
            'error': msg,
            'shape': None,
            'voxel_dims': None,
            'dtype': None
        }
    
    return {
        'path': str(image_path),
        'status': 'success',
        'error': None,
        'shape': img.shape,
        'voxel_dims': img.header.get_zooms()[:3] if hasattr(img.header, 'get_zooms') else None,
        'dtype': str(img.get_data_dtype()),
        'min_intensity': float(np.min(data)) if data is not None else None,
        'max_intensity': float(np.max(data)) if data is not None else None,
        'mean_intensity': float(np.mean(data)) if data is not None else None,
        'std_intensity': float(np.std(data)) if data is not None else None
    }


def find_analyze_pairs(directory: Path) -> list:
    """
    Find all .hdr/.img pairs in a directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of .hdr file paths that have corresponding .img files
    """
    directory = Path(directory)
    hdr_files = list(directory.glob('**/*.hdr'))
    
    valid_pairs = []
    for hdr_file in hdr_files:
        img_file = hdr_file.with_suffix('.img')
        if img_file.exists():
            valid_pairs.append(hdr_file)
    
    return valid_pairs


def batch_convert_analyze_to_nifti(
    directory: Path,
    pattern: str = '**/*.hdr',
    overwrite: bool = False
) -> dict:
    """
    Batch convert all Analyze files in a directory to NIfTI.
    
    Args:
        directory: Root directory to search
        pattern: Glob pattern for .hdr files
        overwrite: Whether to overwrite existing files
        
    Returns:
        Dictionary with conversion statistics
    """
    directory = Path(directory)
    hdr_files = list(directory.glob(pattern))
    
    results = {
        'total': len(hdr_files),
        'success': 0,
        'skipped': 0,
        'failed': 0,
        'errors': []
    }
    
    for hdr_file in hdr_files:
        img_file = hdr_file.with_suffix('.img')
        if not img_file.exists():
            results['failed'] += 1
            results['errors'].append(f"Missing .img for {hdr_file}")
            continue
        
        success, msg = convert_analyze_to_nifti(hdr_file, overwrite=overwrite)
        
        if success:
            if 'skipped' in msg.lower():
                results['skipped'] += 1
            else:
                results['success'] += 1
        else:
            results['failed'] += 1
            results['errors'].append(msg)
    
    return results
