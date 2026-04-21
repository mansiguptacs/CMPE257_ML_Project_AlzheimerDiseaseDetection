"""
Quality control utilities for OASIS-1 MRI data.

Provides functions for:
- Generating middle slice visualizations
- Computing image quality metrics
- Creating QC reports
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pandas as pd

logger = logging.getLogger(__name__)


def get_middle_slices(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract middle slices in all three orientations.
    
    Args:
        data: 3D image array
        
    Returns:
        Tuple of (axial, coronal, sagittal) middle slices
    """
    shape = data.shape
    
    axial = data[:, :, shape[2] // 2]
    coronal = data[:, shape[1] // 2, :]
    sagittal = data[shape[0] // 2, :, :]
    
    return axial, coronal, sagittal


def create_qc_montage(
    image_path: Path,
    output_path: Path,
    title: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Create a 3-panel QC montage showing middle slices.
    
    Args:
        image_path: Path to input image
        output_path: Path to save PNG
        title: Optional title for the figure
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        img = nib.load(str(image_path))
        data = img.get_fdata()
        
        axial, coronal, sagittal = get_middle_slices(data)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Axial
        axes[0].imshow(axial.T, cmap='gray', origin='lower')
        axes[0].set_title('Axial (Middle)')
        axes[0].axis('off')
        
        # Coronal
        axes[1].imshow(coronal.T, cmap='gray', origin='lower')
        axes[1].set_title('Coronal (Middle)')
        axes[1].axis('off')
        
        # Sagittal
        axes[2].imshow(sagittal.T, cmap='gray', origin='lower')
        axes[2].set_title('Sagittal (Middle)')
        axes[2].axis('off')
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return True, f"QC montage saved: {output_path}"
        
    except Exception as e:
        return False, f"QC montage failed: {str(e)}"


def compute_image_stats(data: np.ndarray) -> dict:
    """
    Compute basic statistics for an image.
    
    Args:
        data: 3D image array
        
    Returns:
        Dictionary of statistics
    """
    # Mask out zeros (background)
    nonzero_data = data[data > 0]
    
    if len(nonzero_data) == 0:
        return {
            'min': 0.0,
            'max': 0.0,
            'mean': 0.0,
            'std': 0.0,
            'median': 0.0,
            'p01': 0.0,
            'p99': 0.0,
            'nonzero_voxels': 0
        }
    
    return {
        'min': float(np.min(nonzero_data)),
        'max': float(np.max(nonzero_data)),
        'mean': float(np.mean(nonzero_data)),
        'std': float(np.std(nonzero_data)),
        'median': float(np.median(nonzero_data)),
        'p01': float(np.percentile(nonzero_data, 1)),
        'p99': float(np.percentile(nonzero_data, 99)),
        'nonzero_voxels': int(len(nonzero_data))
    }


def generate_session_qc(
    session_id: str,
    image_paths: dict,
    output_dir: Path
) -> dict:
    """
    Generate QC outputs for a single session.
    
    Args:
        session_id: Session identifier (e.g., 'OAS1_0001_MR1')
        image_paths: Dictionary of image types to paths
        output_dir: Output directory for QC files
        
    Returns:
        Dictionary with QC results
    """
    qc_results = {
        'session_id': session_id,
        'qc_pass': True,
        'errors': []
    }
    
    session_qc_dir = output_dir / session_id
    session_qc_dir.mkdir(parents=True, exist_ok=True)
    
    for image_type, image_path in image_paths.items():
        if image_path is None or not Path(image_path).exists():
            qc_results[f'{image_type}_status'] = 'missing'
            qc_results['errors'].append(f"{image_type} file missing")
            continue
        
        try:
            # Load image
            img = nib.load(str(image_path))
            data = img.get_fdata()
            
            # Store basic info
            qc_results[f'{image_type}_shape'] = str(img.shape)
            qc_results[f'{image_type}_voxel_dims'] = str(img.header.get_zooms()[:3])
            
            # Compute stats
            stats = compute_image_stats(data)
            for stat_name, stat_value in stats.items():
                qc_results[f'{image_type}_{stat_name}'] = stat_value
            
            # Generate montage
            montage_path = session_qc_dir / f'{session_id}_{image_type}_qc.png'
            success, msg = create_qc_montage(
                image_path,
                montage_path,
                title=f'{session_id} - {image_type}'
            )
            
            if success:
                qc_results[f'{image_type}_montage'] = str(montage_path)
                qc_results[f'{image_type}_status'] = 'success'
            else:
                qc_results[f'{image_type}_status'] = 'failed'
                qc_results['errors'].append(msg)
                qc_results['qc_pass'] = False
                
        except Exception as e:
            qc_results[f'{image_type}_status'] = 'error'
            qc_results['errors'].append(f"{image_type}: {str(e)}")
            qc_results['qc_pass'] = False
    
    return qc_results


def create_qc_summary_report(qc_results: list, output_path: Path):
    """
    Create a summary CSV report from QC results.
    
    Args:
        qc_results: List of QC result dictionaries
        output_path: Path to save CSV report
    """
    df = pd.DataFrame(qc_results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"QC summary report saved: {output_path}")
    logger.info(f"Total sessions: {len(df)}")
    logger.info(f"QC pass: {df['qc_pass'].sum()}")
    logger.info(f"QC fail: {(~df['qc_pass']).sum()}")
