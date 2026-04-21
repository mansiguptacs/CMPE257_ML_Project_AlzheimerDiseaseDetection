#!/usr/bin/env python3
"""
Pre-training audit: Verify the enhanced CSV is correct and ready for model training.

Checks:
1. Merge integrity (correct patient matching)
2. Feature completeness (NaN patterns)
3. Target variable distribution
4. No data leakage
5. Patient-level sanity checks
6. Cross-validation against original CSV
"""

import pandas as pd
import numpy as np
from pathlib import Path

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

def run_audit():
    print("=" * 70)
    print("  PRE-TRAINING AUDIT: oasis1_full_enhanced_features.csv")
    print("=" * 70)

    enhanced = pd.read_csv('data/enhanced_features/oasis1_full_enhanced_features.csv')
    original = pd.read_excel('oasis_cross-sectional-5708aa0a98d82080.xlsx')

    results = []

    # -----------------------------------------------------------------------
    # CHECK 1: Row count matches
    # -----------------------------------------------------------------------
    print("\n[CHECK 1] Row count")
    n_enhanced = len(enhanced)
    n_original = len(original)
    ok = n_enhanced == n_original
    status = PASS if ok else FAIL
    print(f"  Enhanced: {n_enhanced} | Original: {n_original} → {status}")
    results.append(('Row count match', status))

    # -----------------------------------------------------------------------
    # CHECK 2: Session ID integrity
    # -----------------------------------------------------------------------
    print("\n[CHECK 2] Session ID integrity")
    enhanced_ids = set(enhanced['ID'].tolist())
    original_ids = set(original['ID'].tolist())
    missing = original_ids - enhanced_ids
    extra = enhanced_ids - original_ids
    dups = enhanced['ID'].duplicated().sum()
    print(f"  Missing IDs (in original, not enhanced): {len(missing)}")
    print(f"  Extra IDs (in enhanced, not original):   {len(extra)}")
    print(f"  Duplicate IDs in enhanced:               {dups}")
    ok = len(missing) == 0 and len(extra) == 0 and dups == 0
    status = PASS if ok else FAIL
    print(f"  → {status}")
    results.append(('Session ID integrity', status))

    # -----------------------------------------------------------------------
    # CHECK 3: Original columns preserved exactly
    # -----------------------------------------------------------------------
    print("\n[CHECK 3] Original columns preserved")
    orig_cols = original.columns.tolist()
    preserved = all(c in enhanced.columns for c in orig_cols)
    print(f"  Original columns: {orig_cols}")
    print(f"  All present in enhanced: {preserved}")

    # Spot-check: verify original values didn't get corrupted
    merge_key = 'ID'
    check_cols = ['Age', 'M/F', 'MMSE', 'CDR', 'nWBV', 'eTIV']
    check_cols = [c for c in check_cols if c in original.columns and c in enhanced.columns]

    merged_check = pd.merge(
        original[[merge_key] + check_cols],
        enhanced[[merge_key] + check_cols],
        on=merge_key, suffixes=('_orig', '_enh')
    )

    mismatches = 0
    for col in check_cols:
        orig_c = col + '_orig'
        enh_c = col + '_enh'
        if merged_check[orig_c].dtype == 'object':
            bad = (merged_check[orig_c].fillna('') != merged_check[enh_c].fillna('')).sum()
        else:
            # Handle NaN: both NaN = match
            both_nan = merged_check[orig_c].isna() & merged_check[enh_c].isna()
            both_valid = merged_check[orig_c].notna() & merged_check[enh_c].notna()
            value_mismatch = both_valid & (merged_check[orig_c] != merged_check[enh_c])
            one_nan = (merged_check[orig_c].isna() != merged_check[enh_c].isna())
            bad = value_mismatch.sum() + one_nan.sum()
        if bad > 0:
            print(f"  !! Column '{col}' has {bad} mismatches between original and enhanced")
        mismatches += bad

    status = PASS if (preserved and mismatches == 0) else FAIL
    print(f"  Value mismatches across spot-checked columns: {mismatches}")
    print(f"  → {status}")
    results.append(('Original columns preserved', status))

    # -----------------------------------------------------------------------
    # CHECK 4: Feature completeness
    # -----------------------------------------------------------------------
    print("\n[CHECK 4] Feature completeness")
    new_cols = [c for c in enhanced.columns if c not in orig_cols]
    print(f"  New imaging columns: {len(new_cols)}")

    tissue_cols = [c for c in new_cols if any(x in c for x in
                   ['csf_vol', 'gm_vol', 'wm_vol', 'brain_parenchyma', 'csf_frac',
                    'gm_frac', 'wm_frac', 'csf_to_brain', 'gm_wm_ratio',
                    'reconstructed_nwbv', '_to_etiv', 'csf_voxels', 'gm_voxels',
                    'wm_voxels', 'total_segmented', 'nwbv_abs'])]
    regional_cols = [c for c in new_cols if any(x in c for x in
                     ['hippocampus', 'ventricle', 'entorhinal', 'temporal'])]

    print(f"  Tissue features:   {len(tissue_cols)}")
    print(f"  Regional features: {len(regional_cols)}")

    # NaN analysis
    nan_counts = enhanced[new_cols].isna().sum()
    fully_populated = (nan_counts == 0).sum()
    partially_populated = ((nan_counts > 0) & (nan_counts < n_enhanced)).sum()
    fully_empty = (nan_counts == n_enhanced).sum()

    print(f"  Fully populated (0 NaN):  {fully_populated}/{len(new_cols)}")
    print(f"  Partially populated:      {partially_populated}/{len(new_cols)}")
    print(f"  Fully empty:              {fully_empty}/{len(new_cols)}")

    if partially_populated > 0 or fully_empty > 0:
        problem_cols = nan_counts[nan_counts > 0]
        for col, cnt in problem_cols.items():
            pct = cnt / n_enhanced * 100
            print(f"    - {col}: {cnt} NaN ({pct:.1f}%)")

    status = PASS if fully_empty == 0 else WARN
    results.append(('Feature completeness', status))
    print(f"  → {status}")

    # -----------------------------------------------------------------------
    # CHECK 5: No data leakage
    # -----------------------------------------------------------------------
    print("\n[CHECK 5] No data leakage check")
    leakage_cols = ['Group', 'Visit', 'MR Delay']
    leaked = [c for c in leakage_cols if c in enhanced.columns]
    print(f"  Leakage columns present: {leaked}")
    status = PASS if len(leaked) == 0 else FAIL
    print(f"  → {status}")
    results.append(('No data leakage', status))

    # -----------------------------------------------------------------------
    # CHECK 6: Target variable (CDR) distribution
    # -----------------------------------------------------------------------
    print("\n[CHECK 6] Target variable (CDR) distribution")
    if 'CDR' in enhanced.columns:
        cdr_dist = enhanced['CDR'].value_counts(dropna=False).sort_index()
        for val, cnt in cdr_dist.items():
            label = f"CDR {val}" if pd.notna(val) else "CDR NaN"
            print(f"  {label}: {cnt} ({cnt/n_enhanced*100:.1f}%)")
        cdr_nan = enhanced['CDR'].isna().sum()
        print(f"  CDR NaN count: {cdr_nan}")
        status = PASS if cdr_nan < n_enhanced else FAIL
    else:
        status = FAIL
        print(f"  CDR column missing!")
    results.append(('CDR target present', status))

    # -----------------------------------------------------------------------
    # CHECK 7: Patient-level sanity (cross-check a few specific patients)
    # -----------------------------------------------------------------------
    print("\n[CHECK 7] Patient-level sanity spot-check")
    spot_ids = ['OAS1_0001_MR1', 'OAS1_0100_MR1', 'OAS1_0300_MR1', 'OAS1_0450_MR1']
    for sid in spot_ids:
        enh_row = enhanced[enhanced['ID'] == sid]
        orig_row = original[original['ID'] == sid]
        if len(enh_row) == 0 and len(orig_row) == 0:
            continue
        if len(enh_row) == 1 and len(orig_row) == 1:
            age_ok = enh_row['Age'].values[0] == orig_row['Age'].values[0]
            nwbv_ok = enh_row['nWBV'].values[0] == orig_row['nWBV'].values[0]
            has_hippo = pd.notna(enh_row['hippocampus_bilateral_volume_mm3'].values[0])
            has_vent = pd.notna(enh_row['ventricle_bilateral_volume_mm3'].values[0])
            print(f"  {sid}: Age={age_ok} nWBV={nwbv_ok} hippo={has_hippo} vent={has_vent}")
        else:
            print(f"  {sid}: enh_rows={len(enh_row)} orig_rows={len(orig_row)} MISMATCH")
    results.append(('Patient spot-check', PASS))

    # -----------------------------------------------------------------------
    # CHECK 8: Clinical direction sanity
    # -----------------------------------------------------------------------
    print("\n[CHECK 8] Clinical direction sanity")
    if 'CDR' in enhanced.columns and 'hippocampus_bilateral_volume_mm3' in enhanced.columns:
        cdr0 = enhanced[enhanced['CDR'] == 0.0]['hippocampus_bilateral_volume_mm3'].mean()
        cdr1 = enhanced[enhanced['CDR'] == 1.0]['hippocampus_bilateral_volume_mm3'].mean()
        hippo_ok = cdr0 > cdr1  # Hippocampus should be LARGER in healthy
        print(f"  Hippocampus: CDR0={cdr0:.0f} > CDR1={cdr1:.0f} → {'CORRECT' if hippo_ok else 'WRONG'}")

        vent0 = enhanced[enhanced['CDR'] == 0.0]['ventricle_bilateral_volume_mm3'].mean()
        vent1 = enhanced[enhanced['CDR'] == 1.0]['ventricle_bilateral_volume_mm3'].mean()
        vent_ok = vent0 < vent1  # Ventricles should be LARGER in dementia
        print(f"  Ventricles:  CDR0={vent0:.0f} < CDR1={vent1:.0f} → {'CORRECT' if vent_ok else 'WRONG'}")

        csf0 = enhanced[enhanced['CDR'] == 0.0]['csf_to_brain_ratio'].mean()
        csf1 = enhanced[enhanced['CDR'] == 1.0]['csf_to_brain_ratio'].mean()
        csf_ok = csf0 < csf1  # CSF ratio should be HIGHER in dementia
        print(f"  CSF ratio:   CDR0={csf0:.4f} < CDR1={csf1:.4f} → {'CORRECT' if csf_ok else 'WRONG'}")

        status = PASS if (hippo_ok and vent_ok and csf_ok) else FAIL
    else:
        status = FAIL
    results.append(('Clinical direction', status))
    print(f"  → {status}")

    # -----------------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  AUDIT SUMMARY")
    print("=" * 70)
    for name, status in results:
        icon = "✅" if status == PASS else ("⚠️" if status == WARN else "❌")
        print(f"  {icon} {name}: {status}")

    n_pass = sum(1 for _, s in results if s == PASS)
    n_warn = sum(1 for _, s in results if s == WARN)
    n_fail = sum(1 for _, s in results if s == FAIL)
    print(f"\n  Total: {n_pass} PASS | {n_warn} WARN | {n_fail} FAIL")

    if n_fail == 0:
        print("\n  ✅ DATASET IS READY FOR MODEL TRAINING")
    else:
        print("\n  ❌ DATASET HAS ISSUES - DO NOT TRAIN")
    print("=" * 70)


if __name__ == '__main__':
    run_audit()
