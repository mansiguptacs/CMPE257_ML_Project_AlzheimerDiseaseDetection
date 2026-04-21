#!/usr/bin/env python3
"""
Phase 2: Train all 8 ML models on enhanced OASIS-1 dataset (with imaging features).

This script:
1. Preprocesses the enhanced CSV using the same OASISPreprocessor as Phase 1
2. Trains all 8 models
3. Compares Phase 2 vs Phase 1 performance
4. Analyzes feature importance (imaging vs traditional)
5. Runs ablation: with MMSE, without MMSE, imaging-only
6. Generates clinical accuracy report
"""

import click
import pandas as pd
import numpy as np
import sys
import time
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessor import OASISPreprocessor
from src.models import MLModel
from src.utils import save_json, print_metrics


MODELS = [
    'random_forest',
    'logistic_regression',
    'svm',
    'xgboost',
    'gradient_boosting',
    'knn',
    'naive_bayes',
    'adaboost'
]


def preprocess_enhanced(input_path, output_dir, test_size=0.2, random_state=42):
    """Preprocess enhanced CSV using same pipeline as Phase 1."""
    df = pd.read_csv(input_path)
    df['dataset'] = 'cross_sectional'
    if 'ID' in df.columns:
        df['Subject_ID'] = df['ID']

    preprocessor = OASISPreprocessor()
    target_col, feature_cols = preprocessor.identify_target_and_features(df)

    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
        df, target_col, feature_cols,
        test_size=test_size, random_state=random_state,
        subject_level_split=False
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(out / 'X_train.csv', index=False)
    X_test.to_csv(out / 'X_test.csv', index=False)
    y_train.to_csv(out / 'y_train.csv', index=False, header=['target'])
    y_test.to_csv(out / 'y_test.csv', index=False, header=['target'])
    preprocessor.save_preprocessor(out)

    metadata = {
        'dataset': 'OASIS-1 Enhanced (Phase 2)',
        'target_column': target_col,
        'final_features': preprocessor.feature_names,
        'train_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'num_features': int(X_train.shape[1]),
        'test_size': test_size,
        'random_state': random_state,
        'class_distribution_train': y_train.value_counts().to_dict(),
        'class_distribution_test': y_test.value_counts().to_dict()
    }
    save_json(metadata, out / 'preprocessing_metadata.json')

    return X_train, X_test, y_train, y_test, preprocessor.feature_names


def train_models(X_train, X_test, y_train, y_test, output_dir, random_state=42):
    """Train all 8 models and return results."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []

    for model_type in MODELS:
        try:
            ml_model = MLModel(model_type=model_type, random_state=random_state)
            ml_model.train(X_train, y_train)
            metrics = ml_model.evaluate(X_test, y_test)

            ml_model.save_model(out / f'{model_type}_model.pkl')
            save_json(metrics, out / f'{model_type}_metrics.json')

            fi = ml_model.get_feature_importance()
            if fi is not None:
                fi.to_csv(out / f'{model_type}_feature_importance.csv', index=False)

            results.append({
                'model': model_type,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'roc_auc': metrics['roc_auc'],
            })
        except Exception as e:
            click.echo(f"  !! {model_type} failed: {e}")
            results.append({
                'model': model_type, 'accuracy': 0, 'precision': 0,
                'recall': 0, 'f1_score': 0, 'roc_auc': 0, 'error': str(e)
            })

    return pd.DataFrame(results)


def load_phase1_results(phase1_model_dir):
    """Load Phase 1 model comparison results."""
    comp_path = Path(phase1_model_dir) / 'model_comparison.csv'
    if comp_path.exists():
        return pd.read_csv(comp_path)
    # Try loading individual metrics
    results = []
    for model_type in MODELS:
        metric_path = Path(phase1_model_dir) / f'{model_type}_metrics.json'
        if metric_path.exists():
            with open(metric_path) as f:
                m = json.load(f)
            results.append({
                'model': model_type,
                'accuracy': m.get('accuracy', 0),
                'precision': m.get('precision', 0),
                'recall': m.get('recall', 0),
                'f1_score': m.get('f1_score', 0),
                'roc_auc': m.get('roc_auc', 0),
            })
    return pd.DataFrame(results) if results else None


@click.command()
@click.option('--enhanced-csv', type=click.Path(exists=True),
              default='data/enhanced_features/oasis1_full_enhanced_features.csv')
@click.option('--phase1-models', type=click.Path(),
              default='models/phase1_oasis1',
              help='Directory with Phase 1 model results')
@click.option('--output-dir', type=click.Path(),
              default='results/phase2')
@click.option('--random-state', type=int, default=42)
def main(enhanced_csv, phase1_models, output_dir, random_state):
    """Phase 2: Train on enhanced features and compare to Phase 1."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    click.echo("=" * 70)
    click.echo("  PHASE 2: MODEL TRAINING ON ENHANCED OASIS-1 FEATURES")
    click.echo("=" * 70)

    # ==================================================================
    # STEP 1: Preprocess enhanced dataset (full features)
    # ==================================================================
    click.echo("\n[STEP 1/6] Preprocessing enhanced dataset (ALL features)...")
    data_dir_full = out / 'data_full'
    X_tr, X_te, y_tr, y_te, feature_names = preprocess_enhanced(
        enhanced_csv, data_dir_full, random_state=random_state
    )
    click.echo(f"  Train: {X_tr.shape[0]} samples x {X_tr.shape[1]} features")
    click.echo(f"  Test:  {X_te.shape[0]} samples x {X_te.shape[1]} features")
    click.echo(f"  Train class dist: {y_tr.value_counts().to_dict()}")
    click.echo(f"  Test class dist:  {y_te.value_counts().to_dict()}")
    click.echo(f"  Features: {feature_names[:10]}...")

    # ==================================================================
    # STEP 2: Train all 8 models on full enhanced features
    # ==================================================================
    click.echo(f"\n[STEP 2/6] Training 8 models on FULL enhanced features...")
    t0 = time.time()
    full_results = train_models(X_tr, X_te, y_tr, y_te, out / 'models_full', random_state)
    full_results.to_csv(out / 'full_model_comparison.csv', index=False)
    click.echo(f"  Time: {time.time()-t0:.1f}s")

    click.echo(f"\n  --- Phase 2 Full Results ---")
    click.echo(f"  {'Model':<22} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'AUC':<8}")
    click.echo(f"  {'-'*62}")
    for _, r in full_results.sort_values('accuracy', ascending=False).iterrows():
        click.echo(f"  {r['model']:<22} {r['accuracy']:<8.4f} {r['precision']:<8.4f} "
                    f"{r['recall']:<8.4f} {r['f1_score']:<8.4f} {r['roc_auc']:<8.4f}")

    # ==================================================================
    # STEP 3: Ablation — WITHOUT MMSE (clinical robustness test)
    # ==================================================================
    click.echo(f"\n[STEP 3/6] Ablation: Training WITHOUT MMSE...")
    df_no_mmse = pd.read_csv(enhanced_csv)
    if 'MMSE' in df_no_mmse.columns:
        df_no_mmse = df_no_mmse.drop(columns=['MMSE'])
    df_no_mmse['dataset'] = 'cross_sectional'
    if 'ID' in df_no_mmse.columns:
        df_no_mmse['Subject_ID'] = df_no_mmse['ID']

    preprocessor_no_mmse = OASISPreprocessor()
    tgt, feat = preprocessor_no_mmse.identify_target_and_features(df_no_mmse)
    X_tr_nm, X_te_nm, y_tr_nm, y_te_nm = preprocessor_no_mmse.preprocess_pipeline(
        df_no_mmse, tgt, feat, random_state=random_state, subject_level_split=False
    )
    click.echo(f"  No-MMSE features: {X_tr_nm.shape[1]}")

    no_mmse_results = train_models(X_tr_nm, X_te_nm, y_tr_nm, y_te_nm,
                                    out / 'models_no_mmse', random_state)
    no_mmse_results.to_csv(out / 'no_mmse_model_comparison.csv', index=False)

    click.echo(f"\n  --- Without MMSE Results ---")
    click.echo(f"  {'Model':<22} {'Acc':<8} {'F1':<8} {'AUC':<8}")
    click.echo(f"  {'-'*46}")
    for _, r in no_mmse_results.sort_values('accuracy', ascending=False).iterrows():
        click.echo(f"  {r['model']:<22} {r['accuracy']:<8.4f} {r['f1_score']:<8.4f} {r['roc_auc']:<8.4f}")

    # ==================================================================
    # STEP 4: Ablation — IMAGING FEATURES ONLY (no MMSE, no nWBV, no eTIV)
    # ==================================================================
    click.echo(f"\n[STEP 4/6] Ablation: Training with IMAGING FEATURES ONLY...")
    df_img_only = pd.read_csv(enhanced_csv)
    # Keep only imaging-derived features + basic demographics (Age, Sex, Educ)
    drop_clinical = ['MMSE', 'nWBV', 'eTIV', 'ASF', 'Delay', 'SES']
    for c in drop_clinical:
        if c in df_img_only.columns:
            df_img_only = df_img_only.drop(columns=[c])
    df_img_only['dataset'] = 'cross_sectional'
    if 'ID' in df_img_only.columns:
        df_img_only['Subject_ID'] = df_img_only['ID']

    preprocessor_img = OASISPreprocessor()
    tgt_i, feat_i = preprocessor_img.identify_target_and_features(df_img_only)
    X_tr_i, X_te_i, y_tr_i, y_te_i = preprocessor_img.preprocess_pipeline(
        df_img_only, tgt_i, feat_i, random_state=random_state, subject_level_split=False
    )
    click.echo(f"  Imaging-only features: {X_tr_i.shape[1]}")

    img_results = train_models(X_tr_i, X_te_i, y_tr_i, y_te_i,
                                out / 'models_imaging_only', random_state)
    img_results.to_csv(out / 'imaging_only_model_comparison.csv', index=False)

    click.echo(f"\n  --- Imaging-Only Results ---")
    click.echo(f"  {'Model':<22} {'Acc':<8} {'F1':<8} {'AUC':<8}")
    click.echo(f"  {'-'*46}")
    for _, r in img_results.sort_values('accuracy', ascending=False).iterrows():
        click.echo(f"  {r['model']:<22} {r['accuracy']:<8.4f} {r['f1_score']:<8.4f} {r['roc_auc']:<8.4f}")

    # ==================================================================
    # STEP 5: Compare Phase 2 vs Phase 1
    # ==================================================================
    click.echo(f"\n[STEP 5/6] Comparing Phase 2 vs Phase 1...")
    phase1_results = load_phase1_results(phase1_models)

    if phase1_results is not None and len(phase1_results) > 0:
        comparison = pd.merge(
            phase1_results[['model', 'accuracy', 'f1_score', 'roc_auc']],
            full_results[['model', 'accuracy', 'f1_score', 'roc_auc']],
            on='model', suffixes=('_phase1', '_phase2')
        )
        comparison['acc_delta'] = comparison['accuracy_phase2'] - comparison['accuracy_phase1']
        comparison['f1_delta'] = comparison['f1_score_phase2'] - comparison['f1_score_phase1']
        comparison['auc_delta'] = comparison['roc_auc_phase2'] - comparison['roc_auc_phase1']

        click.echo(f"\n  --- Phase 1 vs Phase 2 Comparison ---")
        click.echo(f"  {'Model':<22} {'P1 Acc':<9} {'P2 Acc':<9} {'Δ Acc':<9} "
                    f"{'P1 AUC':<9} {'P2 AUC':<9} {'Δ AUC':<9}")
        click.echo(f"  {'-'*76}")
        for _, r in comparison.iterrows():
            delta_acc = f"{r['acc_delta']:+.4f}"
            delta_auc = f"{r['auc_delta']:+.4f}"
            click.echo(f"  {r['model']:<22} {r['accuracy_phase1']:<9.4f} {r['accuracy_phase2']:<9.4f} "
                        f"{delta_acc:<9} {r['roc_auc_phase1']:<9.4f} {r['roc_auc_phase2']:<9.4f} {delta_auc:<9}")

        comparison.to_csv(out / 'phase1_vs_phase2_comparison.csv', index=False)

        avg_acc_p1 = comparison['accuracy_phase1'].mean()
        avg_acc_p2 = comparison['accuracy_phase2'].mean()
        avg_auc_p1 = comparison['roc_auc_phase1'].mean()
        avg_auc_p2 = comparison['roc_auc_phase2'].mean()
        click.echo(f"\n  Average Accuracy: Phase1={avg_acc_p1:.4f} → Phase2={avg_acc_p2:.4f} "
                    f"(Δ={avg_acc_p2-avg_acc_p1:+.4f})")
        click.echo(f"  Average AUC:      Phase1={avg_auc_p1:.4f} → Phase2={avg_auc_p2:.4f} "
                    f"(Δ={avg_auc_p2-avg_auc_p1:+.4f})")
    else:
        click.echo("  !! Phase 1 results not found. Skipping comparison.")

    # ==================================================================
    # STEP 6: Feature importance summary
    # ==================================================================
    click.echo(f"\n[STEP 6/6] Feature importance analysis...")

    # Use Random Forest importance as representative
    fi_path = out / 'models_full' / 'random_forest_feature_importance.csv'
    if fi_path.exists():
        fi = pd.read_csv(fi_path)
        click.echo(f"\n  --- Top 20 Features (Random Forest) ---")
        for i, row in fi.head(20).iterrows():
            feat_name = row['feature']
            imp = row['importance']
            # Classify feature source
            if any(x in feat_name for x in ['hippocampus', 'ventricle', 'entorhinal', 'temporal']):
                source = '[REGIONAL]'
            elif any(x in feat_name for x in ['csf_', 'gm_', 'wm_', 'brain_parenchyma', 'nwbv']):
                source = '[TISSUE]'
            elif feat_name in ['MMSE', 'Age', 'Educ', 'SES', 'M/F', 'eTIV', 'nWBV', 'ASF']:
                source = '[ORIGINAL]'
            else:
                source = '[OTHER]'
            click.echo(f"  {i+1:3d}. {feat_name:<45} {imp:.4f}  {source}")

        # Compute category totals
        original_imp = fi[fi['feature'].isin(['MMSE', 'Age', 'Educ', 'SES', 'eTIV', 'nWBV', 'ASF'])]['importance'].sum()
        tissue_imp = fi[fi['feature'].str.contains('csf_|gm_|wm_|brain_parenchyma|reconstructed|_frac|_ratio|_to_etiv', na=False)]['importance'].sum()
        regional_imp = fi[fi['feature'].str.contains('hippocampus|ventricle|entorhinal|temporal', na=False)]['importance'].sum()
        mmse_imp = fi[fi['feature'] == 'MMSE']['importance'].sum() if 'MMSE' in fi['feature'].values else 0

        total = fi['importance'].sum()
        click.echo(f"\n  --- Feature Importance by Category ---")
        click.echo(f"  Original clinical:  {original_imp:.4f} ({original_imp/total*100:.1f}%)")
        click.echo(f"    of which MMSE:    {mmse_imp:.4f} ({mmse_imp/total*100:.1f}%)")
        click.echo(f"  Tissue features:    {tissue_imp:.4f} ({tissue_imp/total*100:.1f}%)")
        click.echo(f"  Regional features:  {regional_imp:.4f} ({regional_imp/total*100:.1f}%)")
        click.echo(f"  Total imaging:      {(tissue_imp+regional_imp):.4f} ({(tissue_imp+regional_imp)/total*100:.1f}%)")

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    click.echo(f"\n{'=' * 70}")
    click.echo("  PHASE 2 TRAINING COMPLETE")
    click.echo("=" * 70)

    best_full = full_results.sort_values('accuracy', ascending=False).iloc[0]
    best_no_mmse = no_mmse_results.sort_values('accuracy', ascending=False).iloc[0]
    best_img = img_results.sort_values('accuracy', ascending=False).iloc[0]

    click.echo(f"\n  Best model (full features):    {best_full['model']} "
               f"(Acc={best_full['accuracy']:.4f}, AUC={best_full['roc_auc']:.4f})")
    click.echo(f"  Best model (no MMSE):          {best_no_mmse['model']} "
               f"(Acc={best_no_mmse['accuracy']:.4f}, AUC={best_no_mmse['roc_auc']:.4f})")
    click.echo(f"  Best model (imaging only):     {best_img['model']} "
               f"(Acc={best_img['accuracy']:.4f}, AUC={best_img['roc_auc']:.4f})")

    click.echo(f"\n  Output directory: {out}")
    click.echo("=" * 70)


if __name__ == '__main__':
    main()
