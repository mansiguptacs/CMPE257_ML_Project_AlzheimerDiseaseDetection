"""
Session-safe merge utilities for OASIS-1 imaging features.

Provides functions for:
- Validating merge keys
- Enforcing one-to-one merge constraints
- Generating merge audit reports
- Detecting and reporting unmatched rows
"""

import logging
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def validate_merge_keys(
    df: pd.DataFrame,
    key_column: str,
    df_name: str = "DataFrame"
) -> dict:
    """
    Validate merge key uniqueness and completeness.
    
    Args:
        df: DataFrame to validate
        key_column: Name of merge key column
        df_name: Name for reporting
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        'df_name': df_name,
        'total_rows': len(df),
        'key_column': key_column,
        'has_key_column': key_column in df.columns
    }
    
    if not validation['has_key_column']:
        validation['is_valid'] = False
        validation['error'] = f"Key column '{key_column}' not found"
        return validation
    
    keys = df[key_column]
    
    validation['null_keys'] = int(keys.isnull().sum())
    validation['empty_keys'] = int((keys == '').sum() if keys.dtype == 'object' else 0)
    validation['unique_keys'] = int(keys.nunique())
    validation['duplicate_keys'] = validation['total_rows'] - validation['unique_keys']
    
    # Check for duplicates
    if validation['duplicate_keys'] > 0:
        duplicates = keys[keys.duplicated(keep=False)].unique()
        validation['duplicate_values'] = duplicates.tolist()[:10]  # First 10
    else:
        validation['duplicate_values'] = []
    
    # Overall validation
    validation['is_valid'] = (
        validation['null_keys'] == 0 and
        validation['empty_keys'] == 0 and
        validation['duplicate_keys'] == 0
    )
    
    return validation


def safe_merge(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    merge_key: str,
    left_name: str = "left",
    right_name: str = "right",
    how: str = "outer"
) -> Tuple[pd.DataFrame, dict]:
    """
    Perform a safe merge with validation and audit trail.
    
    Args:
        left_df: Left DataFrame (typically tabular CSV)
        right_df: Right DataFrame (typically imaging features)
        merge_key: Column name to merge on
        left_name: Name for left DataFrame (for reporting)
        right_name: Name for right DataFrame (for reporting)
        how: Merge type ('inner', 'outer', 'left', 'right')
        
    Returns:
        Tuple of (merged_df, audit_dict)
    """
    # Validate merge keys
    left_validation = validate_merge_keys(left_df, merge_key, left_name)
    right_validation = validate_merge_keys(right_df, merge_key, right_name)
    
    audit = {
        'merge_key': merge_key,
        'merge_type': how,
        'left_validation': left_validation,
        'right_validation': right_validation
    }
    
    # Check if merge is safe
    if not left_validation['is_valid']:
        logger.error(f"Left DataFrame validation failed: {left_validation.get('error')}")
        if left_validation.get('duplicate_keys', 0) > 0:
            logger.error(f"Duplicate keys in {left_name}: {left_validation['duplicate_values']}")
        raise ValueError(f"Left DataFrame ({left_name}) has invalid merge keys")
    
    if not right_validation['is_valid']:
        logger.error(f"Right DataFrame validation failed: {right_validation.get('error')}")
        if right_validation.get('duplicate_keys', 0) > 0:
            logger.error(f"Duplicate keys in {right_name}: {right_validation['duplicate_values']}")
        raise ValueError(f"Right DataFrame ({right_name}) has invalid merge keys")
    
    # Perform merge
    merged_df = pd.merge(
        left_df,
        right_df,
        on=merge_key,
        how=how,
        indicator=True,
        suffixes=('_left', '_right')
    )
    
    # Analyze merge results
    merge_indicator = merged_df['_merge']
    
    audit['total_merged_rows'] = len(merged_df)
    audit['matched_rows'] = int((merge_indicator == 'both').sum())
    audit['left_only_rows'] = int((merge_indicator == 'left_only').sum())
    audit['right_only_rows'] = int((merge_indicator == 'right_only').sum())
    
    # Get unmatched keys
    audit['left_only_keys'] = merged_df[merge_indicator == 'left_only'][merge_key].tolist()
    audit['right_only_keys'] = merged_df[merge_indicator == 'right_only'][merge_key].tolist()
    
    # Calculate match rate
    audit['match_rate'] = audit['matched_rows'] / max(left_validation['total_rows'], right_validation['total_rows'])
    
    return merged_df, audit


def extract_unmatched_rows(
    merged_df: pd.DataFrame,
    merge_key: str,
    side: str = 'left'
) -> pd.DataFrame:
    """
    Extract unmatched rows from a merged DataFrame.
    
    Args:
        merged_df: Merged DataFrame with '_merge' indicator
        merge_key: Merge key column name
        side: Which side to extract ('left' or 'right')
        
    Returns:
        DataFrame with unmatched rows
    """
    if '_merge' not in merged_df.columns:
        raise ValueError("DataFrame must have '_merge' indicator column")
    
    indicator_value = f'{side}_only'
    unmatched = merged_df[merged_df['_merge'] == indicator_value].copy()
    
    # Drop the merge indicator
    unmatched = unmatched.drop(columns=['_merge'])
    
    # Drop columns that are all NaN (from the other side)
    unmatched = unmatched.dropna(axis=1, how='all')
    
    return unmatched


def create_merge_audit_report(
    audit: dict,
    output_path: Path
):
    """
    Create a human-readable merge audit report.
    
    Args:
        audit: Audit dictionary from safe_merge
        output_path: Path to save report
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MERGE AUDIT REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Merge Key: {audit['merge_key']}\n")
        f.write(f"Merge Type: {audit['merge_type']}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("LEFT DATAFRAME VALIDATION\n")
        f.write("-" * 70 + "\n")
        left_val = audit['left_validation']
        f.write(f"Name: {left_val['df_name']}\n")
        f.write(f"Total rows: {left_val['total_rows']}\n")
        f.write(f"Unique keys: {left_val['unique_keys']}\n")
        f.write(f"Duplicate keys: {left_val['duplicate_keys']}\n")
        f.write(f"Null keys: {left_val['null_keys']}\n")
        f.write(f"Valid: {left_val['is_valid']}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("RIGHT DATAFRAME VALIDATION\n")
        f.write("-" * 70 + "\n")
        right_val = audit['right_validation']
        f.write(f"Name: {right_val['df_name']}\n")
        f.write(f"Total rows: {right_val['total_rows']}\n")
        f.write(f"Unique keys: {right_val['unique_keys']}\n")
        f.write(f"Duplicate keys: {right_val['duplicate_keys']}\n")
        f.write(f"Null keys: {right_val['null_keys']}\n")
        f.write(f"Valid: {right_val['is_valid']}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("MERGE RESULTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total merged rows: {audit['total_merged_rows']}\n")
        f.write(f"Matched rows (both): {audit['matched_rows']}\n")
        f.write(f"Left only rows: {audit['left_only_rows']}\n")
        f.write(f"Right only rows: {audit['right_only_rows']}\n")
        f.write(f"Match rate: {audit['match_rate']:.2%}\n\n")
        
        if audit['left_only_rows'] > 0:
            f.write("-" * 70 + "\n")
            f.write("UNMATCHED LEFT KEYS\n")
            f.write("-" * 70 + "\n")
            for key in audit['left_only_keys'][:20]:  # First 20
                f.write(f"  - {key}\n")
            if len(audit['left_only_keys']) > 20:
                f.write(f"  ... and {len(audit['left_only_keys']) - 20} more\n")
            f.write("\n")
        
        if audit['right_only_rows'] > 0:
            f.write("-" * 70 + "\n")
            f.write("UNMATCHED RIGHT KEYS\n")
            f.write("-" * 70 + "\n")
            for key in audit['right_only_keys'][:20]:  # First 20
                f.write(f"  - {key}\n")
            if len(audit['right_only_keys']) > 20:
                f.write(f"  ... and {len(audit['right_only_keys']) - 20} more\n")
            f.write("\n")
        
        f.write("=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")
    
    logger.info(f"Merge audit report saved: {output_path}")


def verify_sample_merge(
    merged_df: pd.DataFrame,
    merge_key: str,
    n_samples: int = 5
) -> pd.DataFrame:
    """
    Extract sample rows to verify merge integrity.
    
    Args:
        merged_df: Merged DataFrame
        merge_key: Merge key column
        n_samples: Number of samples to extract
        
    Returns:
        DataFrame with sample rows
    """
    # Get successfully merged rows
    if '_merge' in merged_df.columns:
        matched = merged_df[merged_df['_merge'] == 'both'].copy()
    else:
        matched = merged_df.copy()
    
    if len(matched) == 0:
        logger.warning("No matched rows found for sample verification")
        return pd.DataFrame()
    
    # Sample rows
    n_samples = min(n_samples, len(matched))
    samples = matched.sample(n=n_samples, random_state=42)
    
    return samples
