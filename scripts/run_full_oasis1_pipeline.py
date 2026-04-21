#!/usr/bin/env python3
"""
Full-scale OASIS-1 MRI Feature Extraction Pipeline.

Processes ALL 12 discs (416+ sessions):
1. Builds unified imaging manifest across all discs
2. Extracts tissue composition features (GM/WM/CSF)
3. Extracts regional ROI features (hippocampus, ventricles, temporal)
4. Merges with tabular CSV
5. Validates clinical plausibility
6. Produces final enhanced CSV
"""

import click
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import re
import time
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.imaging.tissue_features import (
    parse_fsl_seg_txt,
    extract_tissue_voxel_counts
)
from src.imaging.regional_features import (
    extract_session_regional_features_v2,
    validate_regional_features
)

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Unified manifest builder
# ---------------------------------------------------------------------------

def discover_session_files(session_dir: Path) -> dict:
    """Discover key files for a single session directory."""
    session_id = session_dir.name
    record = {
        'session_id': session_id,
        'disc': session_dir.parent.name,
        'session_dir': str(session_dir),
        'path_fsl_seg_hdr': None,
        'path_fsl_seg_txt': None,
        'path_t88_masked_hdr': None,
        'has_fsl_seg_img': False,
        'has_fsl_seg_txt': False,
        'has_t88_masked': False,
    }

    # FSL_SEG files (pattern: *_fseg.hdr / *_fseg.txt)
    fsl_dir = session_dir / 'FSL_SEG'
    if fsl_dir.is_dir():
        for f in fsl_dir.iterdir():
            if f.name.endswith('_fseg.hdr'):
                record['path_fsl_seg_hdr'] = str(f)
                record['has_fsl_seg_img'] = True
            elif f.name.endswith('_fseg.txt'):
                record['path_fsl_seg_txt'] = str(f)
                record['has_fsl_seg_txt'] = True

    # T88 masked image
    t88_dir = session_dir / 'PROCESSED' / 'MPRAGE' / 'T88_111'
    if t88_dir.is_dir():
        for f in t88_dir.iterdir():
            if f.name.endswith('_t88_masked_gfc.hdr'):
                record['path_t88_masked_hdr'] = str(f)
                record['has_t88_masked'] = True
                break

    return record


def build_full_manifest(base_dir: Path) -> pd.DataFrame:
    """Build manifest across all 12 discs."""
    records = []
    for disc_num in range(1, 13):
        disc_dir = base_dir / f'oasis1-disc{disc_num}'
        if not disc_dir.is_dir():
            continue
        for session_dir in sorted(disc_dir.iterdir()):
            if session_dir.is_dir() and session_dir.name.startswith('OAS1_'):
                records.append(discover_session_files(session_dir))
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Step 2: Tissue feature extraction
# ---------------------------------------------------------------------------

def extract_tissue_features_for_session(row: pd.Series, csv_lookup: dict) -> dict:
    """Extract tissue features for one session."""
    session_id = row['session_id']
    features = {'session_id': session_id, 'tissue_status': 'pending'}

    # Parse FSL_SEG txt file for volumes
    txt_path = row.get('path_fsl_seg_txt')
    if txt_path and Path(txt_path).exists():
        txt_result = parse_fsl_seg_txt(txt_path)
        if txt_result.get('status') == 'success':
            features['csf_vol_mm3'] = txt_result.get('csf_vol_mm3', np.nan)
            features['gm_vol_mm3'] = txt_result.get('gm_vol_mm3', np.nan)
            features['wm_vol_mm3'] = txt_result.get('wm_vol_mm3', np.nan)
        else:
            features['csf_vol_mm3'] = np.nan
            features['gm_vol_mm3'] = np.nan
            features['wm_vol_mm3'] = np.nan
    else:
        features['csf_vol_mm3'] = np.nan
        features['gm_vol_mm3'] = np.nan
        features['wm_vol_mm3'] = np.nan

    # Extract voxel counts from segmentation image
    seg_path = row.get('path_fsl_seg_hdr')
    if seg_path and Path(seg_path).exists():
        import nibabel as nib
        try:
            img = nib.load(seg_path)
            data = img.get_fdata().squeeze()
            # FSL FAST: 1=CSF, 2=GM, 3=WM, 0=background
            features['csf_voxels'] = int(np.sum(data == 1))
            features['gm_voxels'] = int(np.sum(data == 2))
            features['wm_voxels'] = int(np.sum(data == 3))
        except Exception:
            features['csf_voxels'] = np.nan
            features['gm_voxels'] = np.nan
            features['wm_voxels'] = np.nan
    else:
        features['csf_voxels'] = np.nan
        features['gm_voxels'] = np.nan
        features['wm_voxels'] = np.nan

    # Derived metrics
    csf = features.get('csf_vol_mm3', np.nan)
    gm = features.get('gm_vol_mm3', np.nan)
    wm = features.get('wm_vol_mm3', np.nan)

    if not (np.isnan(csf) or np.isnan(gm) or np.isnan(wm)):
        total = csf + gm + wm
        brain = gm + wm
        features['brain_parenchyma_vol_mm3'] = brain
        features['total_segmented_vol_mm3'] = total

        if total > 0:
            features['csf_frac'] = csf / total
            features['gm_frac'] = gm / total
            features['wm_frac'] = wm / total
            features['brain_parenchyma_frac'] = brain / total
        else:
            features['csf_frac'] = np.nan
            features['gm_frac'] = np.nan
            features['wm_frac'] = np.nan
            features['brain_parenchyma_frac'] = np.nan

        if brain > 0:
            features['csf_to_brain_ratio'] = csf / brain
        else:
            features['csf_to_brain_ratio'] = np.nan

        if wm > 0:
            features['gm_wm_ratio'] = gm / wm
        else:
            features['gm_wm_ratio'] = np.nan

        # Reconstructed nWBV
        features['reconstructed_nwbv'] = features.get('brain_parenchyma_frac', np.nan)

        # eTIV normalization
        csv_data = csv_lookup.get(session_id, {})
        etiv = csv_data.get('eTIV')
        if etiv and etiv > 0:
            features['brain_parenchyma_to_etiv'] = brain / etiv
            features['gm_to_etiv'] = gm / etiv
            features['wm_to_etiv'] = wm / etiv
            features['csf_to_etiv'] = csf / etiv

        # nWBV validation
        csv_nwbv = csv_data.get('nWBV')
        if csv_nwbv and not np.isnan(features.get('reconstructed_nwbv', np.nan)):
            features['nwbv_abs_error'] = abs(features['reconstructed_nwbv'] - csv_nwbv)

        features['tissue_status'] = 'success'
    else:
        features['tissue_status'] = 'failed'

    return features


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

@click.command()
@click.option('--base-dir', type=click.Path(exists=True), default='.', help='Project base directory')
@click.option('--csv', type=click.Path(exists=True), default='oasis_cross-sectional-5708aa0a98d82080.xlsx')
@click.option('--output-dir', type=click.Path(), default='data/enhanced_features')
def run_pipeline(base_dir, csv, output_dir):
    """Run full OASIS-1 feature extraction pipeline across all 12 discs."""

    base = Path(base_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    click.echo("=" * 70)
    click.echo("  OASIS-1 FULL-SCALE FEATURE EXTRACTION PIPELINE")
    click.echo("  Processing all 12 discs → 416 sessions")
    click.echo("=" * 70)

    # -----------------------------------------------------------------------
    # STEP 1: Build unified manifest
    # -----------------------------------------------------------------------
    click.echo("\n[STEP 1/6] Building unified manifest across all 12 discs...")
    t0 = time.time()
    manifest = build_full_manifest(base)
    manifest_path = out / 'full_imaging_manifest.csv'
    manifest.to_csv(manifest_path, index=False)

    n_total = len(manifest)
    n_fsl = manifest['has_fsl_seg_img'].sum()
    n_txt = manifest['has_fsl_seg_txt'].sum()
    n_t88 = manifest['has_t88_masked'].sum()
    click.echo(f"  Sessions discovered: {n_total}")
    click.echo(f"  FSL_SEG images:      {n_fsl}/{n_total}")
    click.echo(f"  FSL_SEG txt files:   {n_txt}/{n_total}")
    click.echo(f"  T88 masked images:   {n_t88}/{n_total}")
    click.echo(f"  Manifest saved:      {manifest_path}")
    click.echo(f"  Time: {time.time()-t0:.1f}s")

    # -----------------------------------------------------------------------
    # STEP 2: Load tabular CSV
    # -----------------------------------------------------------------------
    click.echo(f"\n[STEP 2/6] Loading tabular CSV...")
    csv_df = pd.read_excel(csv)
    csv_ids = set(csv_df['ID'].tolist())
    click.echo(f"  CSV rows: {len(csv_df)}")

    csv_lookup = {}
    for _, row in csv_df.iterrows():
        csv_lookup[row['ID']] = {
            'eTIV': row.get('eTIV'),
            'nWBV': row.get('nWBV'),
            'CDR': row.get('CDR'),
            'MMSE': row.get('MMSE'),
        }

    # Filter manifest to only sessions in CSV (skip MR2 extras)
    manifest_matched = manifest[manifest['session_id'].isin(csv_ids)].copy()
    click.echo(f"  Matched to CSV: {len(manifest_matched)}/{n_total} sessions")

    unmatched_img = manifest[~manifest['session_id'].isin(csv_ids)]
    if len(unmatched_img) > 0:
        click.echo(f"  Skipping {len(unmatched_img)} imaging-only sessions (MR2 repeats)")

    # -----------------------------------------------------------------------
    # STEP 3: Extract tissue features
    # -----------------------------------------------------------------------
    click.echo(f"\n[STEP 3/6] Extracting tissue features ({len(manifest_matched)} sessions)...")
    t0 = time.time()
    tissue_results = []
    for _, row in tqdm(manifest_matched.iterrows(), total=len(manifest_matched), desc="  Tissue"):
        tissue_results.append(extract_tissue_features_for_session(row, csv_lookup))
    tissue_df = pd.DataFrame(tissue_results)

    n_tissue_ok = (tissue_df['tissue_status'] == 'success').sum()
    n_tissue_fail = (tissue_df['tissue_status'] == 'failed').sum()
    click.echo(f"  Success: {n_tissue_ok} | Failed: {n_tissue_fail}")

    # nWBV validation
    if 'nwbv_abs_error' in tissue_df.columns:
        valid_errs = tissue_df['nwbv_abs_error'].dropna()
        if len(valid_errs) > 0:
            click.echo(f"  nWBV validation (n={len(valid_errs)}):")
            click.echo(f"    Mean error: {valid_errs.mean():.6f}")
            click.echo(f"    Max error:  {valid_errs.max():.6f}")
            click.echo(f"    All < 0.01: {(valid_errs < 0.01).all()}")

    tissue_path = out / 'full_tissue_features.csv'
    tissue_df.to_csv(tissue_path, index=False)
    click.echo(f"  Saved: {tissue_path}")
    click.echo(f"  Time: {time.time()-t0:.1f}s")

    # -----------------------------------------------------------------------
    # STEP 4: Extract regional features
    # -----------------------------------------------------------------------
    click.echo(f"\n[STEP 4/6] Extracting regional features ({len(manifest_matched)} sessions)...")
    click.echo(f"  Method: Tissue-specific volumes within anatomical ROIs")
    click.echo(f"  ROIs: hippocampus(GM), ventricles(CSF), entorhinal(GM), temporal(GM)")
    t0 = time.time()

    regional_results = []
    for _, row in tqdm(manifest_matched.iterrows(), total=len(manifest_matched), desc="  Regional"):
        session_id = row['session_id']
        fsl_seg_path = Path(row['path_fsl_seg_hdr']) if pd.notna(row.get('path_fsl_seg_hdr')) else None
        csv_etiv = csv_lookup.get(session_id, {}).get('eTIV')

        features = extract_session_regional_features_v2(
            session_id=session_id,
            fsl_seg_image_path=fsl_seg_path,
            csv_etiv=csv_etiv
        )
        regional_results.append(features)

    regional_df = pd.DataFrame(regional_results)

    validation = validate_regional_features(regional_df)
    click.echo(f"  Success: {validation['successful_extractions']} | "
               f"Partial: {validation['partial_extractions']} | "
               f"Failed: {validation['failed_extractions']}")

    if 'hippocampus_volume_stats' in validation:
        s = validation['hippocampus_volume_stats']
        click.echo(f"  Hippocampus: mean={s['mean']:.0f} std={s['std']:.0f} "
                    f"CV={s['cv']:.3f} range=[{s['min']:.0f},{s['max']:.0f}]")
    if 'ventricle_volume_stats' in validation:
        s = validation['ventricle_volume_stats']
        click.echo(f"  Ventricles:  mean={s['mean']:.0f} std={s['std']:.0f} "
                    f"CV={s['cv']:.3f} range=[{s['min']:.0f},{s['max']:.0f}]")

    regional_path = out / 'full_regional_features.csv'
    regional_df.to_csv(regional_path, index=False)
    click.echo(f"  Saved: {regional_path}")
    click.echo(f"  Time: {time.time()-t0:.1f}s")

    # -----------------------------------------------------------------------
    # STEP 5: Merge everything
    # -----------------------------------------------------------------------
    click.echo(f"\n[STEP 5/6] Merging tissue + regional + tabular CSV...")

    # Combine tissue + regional on session_id
    drop_cols_tissue = ['tissue_status']
    drop_cols_regional = ['canonical_session_key', 'regional_extraction_status',
                          'regional_extraction_error', 'status', 'error']

    tissue_clean = tissue_df.drop(columns=[c for c in drop_cols_tissue if c in tissue_df.columns])
    regional_clean = regional_df.drop(columns=[c for c in drop_cols_regional if c in regional_df.columns])

    imaging_merged = pd.merge(tissue_clean, regional_clean, on='session_id', how='inner')
    click.echo(f"  Imaging features combined: {len(imaging_merged)} sessions x {len(imaging_merged.columns)} cols")

    # Merge with CSV
    imaging_merged = imaging_merged.rename(columns={'session_id': 'ID'})
    final = pd.merge(csv_df, imaging_merged, on='ID', how='inner')

    click.echo(f"  Final merged: {len(final)} sessions x {len(final.columns)} columns")

    # Validate merge
    assert len(final) == len(imaging_merged), \
        f"Merge mismatch: {len(final)} != {len(imaging_merged)}"
    assert final['ID'].nunique() == len(final), \
        f"Duplicate IDs in final: {final['ID'].nunique()} unique out of {len(final)}"

    final_path = out / 'oasis1_full_enhanced_features.csv'
    final.to_csv(final_path, index=False)
    click.echo(f"  Saved: {final_path}")

    # -----------------------------------------------------------------------
    # STEP 6: Clinical validation
    # -----------------------------------------------------------------------
    click.echo(f"\n[STEP 6/6] Clinical plausibility validation...")

    # CDR group comparison
    if 'CDR' in final.columns and 'hippocampus_bilateral_volume_mm3' in final.columns:
        click.echo(f"\n  --- Hippocampal volume by CDR group ---")
        for cdr_val in sorted(final['CDR'].dropna().unique()):
            grp = final[final['CDR'] == cdr_val]
            hippo = grp['hippocampus_bilateral_volume_mm3'].dropna()
            if len(hippo) > 0:
                click.echo(f"  CDR {cdr_val}: n={len(hippo):3d}  "
                            f"hippo={hippo.mean():8.0f} +/- {hippo.std():7.0f} mm3")

    if 'CDR' in final.columns and 'ventricle_bilateral_volume_mm3' in final.columns:
        click.echo(f"\n  --- Ventricular volume by CDR group ---")
        for cdr_val in sorted(final['CDR'].dropna().unique()):
            grp = final[final['CDR'] == cdr_val]
            vent = grp['ventricle_bilateral_volume_mm3'].dropna()
            if len(vent) > 0:
                click.echo(f"  CDR {cdr_val}: n={len(vent):3d}  "
                            f"vent={vent.mean():8.0f} +/- {vent.std():7.0f} mm3")

    if 'CDR' in final.columns and 'csf_to_brain_ratio' in final.columns:
        click.echo(f"\n  --- CSF-to-brain ratio by CDR group ---")
        for cdr_val in sorted(final['CDR'].dropna().unique()):
            grp = final[final['CDR'] == cdr_val]
            ratio = grp['csf_to_brain_ratio'].dropna()
            if len(ratio) > 0:
                click.echo(f"  CDR {cdr_val}: n={len(ratio):3d}  "
                            f"ratio={ratio.mean():.4f} +/- {ratio.std():.4f}")

    # Correlation with MMSE
    corr_targets = ['hippocampus_bilateral_volume_mm3', 'ventricle_bilateral_volume_mm3',
                     'csf_to_brain_ratio', 'gm_vol_mm3', 'brain_parenchyma_frac']
    corr_available = [c for c in corr_targets if c in final.columns]
    if 'MMSE' in final.columns and len(corr_available) > 0:
        valid = final[['MMSE'] + corr_available].dropna()
        if len(valid) > 5:
            click.echo(f"\n  --- Correlations with MMSE (n={len(valid)}) ---")
            for col in corr_available:
                r = valid['MMSE'].corr(valid[col])
                click.echo(f"  MMSE vs {col}: r = {r:.4f}")

    # Age correlation
    if 'Age' in final.columns and 'ventricle_bilateral_volume_mm3' in final.columns:
        valid_age = final[['Age', 'ventricle_bilateral_volume_mm3',
                            'hippocampus_bilateral_volume_mm3']].dropna()
        if len(valid_age) > 5:
            click.echo(f"\n  --- Correlations with Age (n={len(valid_age)}) ---")
            for col in ['ventricle_bilateral_volume_mm3', 'hippocampus_bilateral_volume_mm3']:
                r = valid_age['Age'].corr(valid_age[col])
                click.echo(f"  Age vs {col}: r = {r:.4f}")

    # Summary
    click.echo(f"\n{'=' * 70}")
    click.echo("  PIPELINE COMPLETE")
    click.echo("=" * 70)
    click.echo(f"  Discs processed:     12")
    click.echo(f"  Sessions discovered: {n_total}")
    click.echo(f"  Sessions matched:    {len(manifest_matched)}")
    click.echo(f"  Tissue success:      {n_tissue_ok}/{len(manifest_matched)}")
    click.echo(f"  Regional success:    {validation['successful_extractions']}/{len(manifest_matched)}")
    click.echo(f"  Final CSV:           {len(final)} rows x {len(final.columns)} columns")
    click.echo(f"  Output:              {final_path}")
    click.echo("=" * 70)


if __name__ == '__main__':
    run_pipeline()
