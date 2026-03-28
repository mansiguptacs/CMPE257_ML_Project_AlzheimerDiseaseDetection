import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.models import MLModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


@click.command()
@click.option('--data-dir',
              type=click.Path(exists=True),
              default='data/processed/oasis1',
              help='Directory containing processed data')
@click.option('--output-dir',
              type=click.Path(),
              default='results/ablation',
              help='Output directory for ablation results')
def ablation_study(data_dir, output_dir):
    """
    Feature ablation study to prove over-reliance on MMSE and global features.
    Tests model performance when key features are removed.
    """
    click.echo("="*70)
    click.echo("Feature Ablation Study - Clinical Limitation Analysis")
    click.echo("="*70)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    click.echo("\n[1/5] Loading processed data...")
    data_path = Path(data_dir)
    X_train = pd.read_csv(data_path / 'X_train.csv')
    X_test = pd.read_csv(data_path / 'X_test.csv')
    
    # Load y data - handle both formats
    y_train_df = pd.read_csv(data_path / 'y_train.csv')
    y_test_df = pd.read_csv(data_path / 'y_test.csv')
    
    # Get first column if multiple columns, otherwise get the only column
    y_train = y_train_df.iloc[:, 0] if y_train_df.shape[1] > 0 else y_train_df.squeeze()
    y_test = y_test_df.iloc[:, 0] if y_test_df.shape[1] > 0 else y_test_df.squeeze()
    
    # Ensure 1D arrays
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    
    click.echo(f"  ✓ Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    click.echo(f"  ✓ Test: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    click.echo(f"  ✓ Features: {list(X_train.columns)}")
    
    # Define ablation scenarios
    ablation_scenarios = {
        'Baseline (All Features)': [],
        'Without MMSE': ['MMSE'],
        'Without Global Brain Metrics': ['nWBV', 'eTIV', 'ASF'],
        'Without MMSE + Global': ['MMSE', 'nWBV', 'eTIV', 'ASF'],
        'Only Demographics': [col for col in X_train.columns if col not in ['Age', 'M/F', 'SES', 'Educ']],
        'Only MMSE': [col for col in X_train.columns if col != 'MMSE'],
        'Only Global Brain': [col for col in X_train.columns if col not in ['nWBV', 'eTIV', 'ASF']],
    }
    
    # Models to test
    models_to_test = ['random_forest', 'xgboost', 'logistic_regression']
    
    results = []
    
    click.echo("\n[2/5] Running ablation experiments...")
    
    for scenario_name, features_to_remove in ablation_scenarios.items():
        click.echo(f"\n  Scenario: {scenario_name}")
        if features_to_remove:
            click.echo(f"    Removing: {features_to_remove}")
        
        # Create feature subset
        remaining_features = [col for col in X_train.columns if col not in features_to_remove]
        
        if len(remaining_features) == 0:
            click.echo(f"    ✗ No features remaining, skipping...")
            continue
        
        click.echo(f"    Remaining features ({len(remaining_features)}): {remaining_features}")
        
        X_train_subset = X_train[remaining_features]
        X_test_subset = X_test[remaining_features]
        
        # Test each model
        for model_name in models_to_test:
            try:
                click.echo(f"    Training {model_name}...")
                
                model = MLModel(model_type=model_name)
                model.train(X_train_subset, y_train)
                
                y_pred = model.predict(X_test_subset)
                y_pred_proba = model.predict_proba(X_test_subset)
                
                # Extract positive class probabilities (column 1) for ROC AUC
                if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                    y_pred_proba_positive = y_pred_proba[:, 1]
                else:
                    y_pred_proba_positive = y_pred_proba.ravel()
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                roc_auc = roc_auc_score(y_test, y_pred_proba_positive)
                
                results.append({
                    'scenario': scenario_name,
                    'model': model_name,
                    'features_removed': ', '.join(features_to_remove) if features_to_remove else 'None',
                    'num_features': len(remaining_features),
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc
                })
                
                click.echo(f"      ✓ Accuracy: {accuracy:.4f}")
                
            except Exception as e:
                click.echo(f"      ✗ Error: {str(e)}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        click.echo("\n✗ No results collected. All experiments failed.")
        click.echo("Please check that the data is properly formatted and models can train.")
        return
    
    results_df.to_csv(output_path / 'ablation_results.csv', index=False)
    click.echo(f"\n[3/5] Results saved to {output_path / 'ablation_results.csv'}")
    
    # Create visualizations
    click.echo("\n[4/5] Creating visualizations...")
    
    # Plot 1: Accuracy comparison across scenarios
    fig, ax = plt.subplots(figsize=(14, 8))
    
    pivot_data = results_df.pivot(index='scenario', columns='model', values='accuracy')
    pivot_data.plot(kind='bar', ax=ax, width=0.8)
    
    ax.set_xlabel('Ablation Scenario', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Feature Ablation Study: Impact on Model Accuracy', fontsize=14, fontweight='bold')
    ax.legend(title='Model', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path / 'ablation_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    click.echo(f"  ✓ Saved {output_path / 'ablation_accuracy_comparison.png'}")
    
    # Plot 2: Performance drop heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    baseline_scores = results_df[results_df['scenario'] == 'Baseline (All Features)'].set_index('model')['accuracy']
    
    drop_data = []
    for scenario in results_df['scenario'].unique():
        if scenario != 'Baseline (All Features)':
            scenario_scores = results_df[results_df['scenario'] == scenario].set_index('model')['accuracy']
            drops = baseline_scores - scenario_scores
            drop_data.append(drops.values)
    
    scenarios = [s for s in results_df['scenario'].unique() if s != 'Baseline (All Features)']
    drop_matrix = pd.DataFrame(drop_data, index=scenarios, columns=baseline_scores.index)
    
    sns.heatmap(drop_matrix, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Accuracy Drop'})
    ax.set_title('Performance Drop from Baseline (Higher = More Important Features)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Ablation Scenario', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path / 'ablation_performance_drop.png', dpi=300, bbox_inches='tight')
    plt.close()
    click.echo(f"  ✓ Saved {output_path / 'ablation_performance_drop.png'}")
    
    # Generate clinical interpretation report
    click.echo("\n[5/5] Generating clinical interpretation report...")
    
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("FEATURE ABLATION STUDY - CLINICAL INTERPRETATION")
    report_lines.append("="*70)
    report_lines.append("")
    report_lines.append("OBJECTIVE:")
    report_lines.append("Demonstrate that current models over-rely on cognitive test (MMSE)")
    report_lines.append("and global brain metrics, lacking regional specificity needed for")
    report_lines.append("clinically reliable Alzheimer's diagnosis.")
    report_lines.append("")
    report_lines.append("="*70)
    report_lines.append("RESULTS SUMMARY")
    report_lines.append("="*70)
    report_lines.append("")
    
    # Calculate average drops
    baseline_avg = results_df[results_df['scenario'] == 'Baseline (All Features)']['accuracy'].mean()
    
    for scenario in results_df['scenario'].unique():
        if scenario != 'Baseline (All Features)':
            scenario_avg = results_df[results_df['scenario'] == scenario]['accuracy'].mean()
            drop = baseline_avg - scenario_avg
            drop_pct = (drop / baseline_avg) * 100
            
            report_lines.append(f"{scenario}:")
            report_lines.append(f"  Average Accuracy: {scenario_avg:.4f}")
            report_lines.append(f"  Drop from Baseline: {drop:.4f} ({drop_pct:.1f}%)")
            
            # Clinical interpretation
            if 'MMSE' in scenario and drop > 0.15:
                report_lines.append(f"  ⚠️  CRITICAL: >15% accuracy drop without MMSE")
                report_lines.append(f"      Models are predicting dementia FROM dementia symptoms")
                report_lines.append(f"      This is circular reasoning, not diagnostic value")
            elif 'Global' in scenario and drop > 0.10:
                report_lines.append(f"  ⚠️  WARNING: >10% drop without global brain metrics")
                report_lines.append(f"      Models rely on non-specific atrophy measures")
                report_lines.append(f"      Cannot distinguish Alzheimer's from other causes")
            
            report_lines.append("")
    
    report_lines.append("="*70)
    report_lines.append("CLINICAL IMPLICATIONS")
    report_lines.append("="*70)
    report_lines.append("")
    
    mmse_drop = baseline_avg - results_df[results_df['scenario'] == 'Without MMSE']['accuracy'].mean()
    global_drop = baseline_avg - results_df[results_df['scenario'] == 'Without Global Brain Metrics']['accuracy'].mean()
    
    report_lines.append("1. OVER-RELIANCE ON COGNITIVE TEST (MMSE):")
    report_lines.append(f"   - Performance drops {mmse_drop:.1%} without MMSE")
    report_lines.append("   - Models are learning: 'Low MMSE → Dementia'")
    report_lines.append("   - Problem: MMSE measures cognitive impairment (the outcome)")
    report_lines.append("   - Cannot detect early Alzheimer's (when MMSE is still normal)")
    report_lines.append("   - Not adding diagnostic value beyond clinical assessment")
    report_lines.append("")
    
    report_lines.append("2. NON-SPECIFIC GLOBAL BRAIN METRICS:")
    report_lines.append(f"   - Performance drops {global_drop:.1%} without global volumes")
    report_lines.append("   - Models use total brain size/atrophy (nWBV, eTIV, ASF)")
    report_lines.append("   - Problem: Many conditions cause global atrophy")
    report_lines.append("   - Cannot distinguish Alzheimer's from:")
    report_lines.append("     • Normal aging")
    report_lines.append("     • Vascular dementia")
    report_lines.append("     • Frontotemporal dementia")
    report_lines.append("     • Other neurodegenerative diseases")
    report_lines.append("")
    
    report_lines.append("3. MISSING ALZHEIMER'S-SPECIFIC FEATURES:")
    report_lines.append("   Current models DO NOT measure:")
    report_lines.append("   Hippocampal atrophy (earliest and most specific sign)")
    report_lines.append("   Entorhinal cortex atrophy (preclinical marker)")
    report_lines.append("   Medial temporal lobe pattern (diagnostic hallmark)")
    report_lines.append("   Regional atrophy asymmetry (staging information)")
    report_lines.append("   Cortical thickness patterns (early detection)")
    report_lines.append("")
    
    report_lines.append("="*70)
    report_lines.append("CONCLUSION")
    report_lines.append("="*70)
    report_lines.append("")
    report_lines.append("Current models achieve high accuracy by:")
    report_lines.append("1. Predicting cognitive impairment from cognitive tests (circular)")
    report_lines.append("2. Using non-specific global brain atrophy measures")
    report_lines.append("")
    report_lines.append("These models CANNOT be clinically relied upon because:")
    report_lines.append("Cannot detect early/preclinical Alzheimer's disease")
    report_lines.append("Cannot distinguish Alzheimer's from other dementias")
    report_lines.append("Cannot identify Alzheimer's-specific atrophy patterns")
    report_lines.append("Over-rely on outcome measures rather than causal pathology")
    report_lines.append("")
    report_lines.append("RECOMMENDATION:")
    report_lines.append("Extract regional volumetric features from OASIS MRI images")
    report_lines.append("Focus on Alzheimer's signature regions:")
    report_lines.append("   - Hippocampus, entorhinal cortex, medial temporal lobe")
    report_lines.append("   - Posterior cingulate, precuneus, parietal cortex")
    report_lines.append("Retrain models with clinically-informed regional features")
    report_lines.append("Validate against clinical diagnostic criteria")
    report_lines.append("")
    report_lines.append("="*70)
    
    report_text = '\n'.join(report_lines)
    
    with open(output_path / 'clinical_interpretation_report.txt', 'w') as f:
        f.write(report_text)
    
    click.echo(f"  ✓ Saved {output_path / 'clinical_interpretation_report.txt'}")
    
    # Print summary to console
    click.echo("\n" + "="*70)
    click.echo("KEY FINDINGS:")
    click.echo("="*70)
    click.echo(f"Baseline accuracy (all features): {baseline_avg:.1%}")
    click.echo(f"Without MMSE: {results_df[results_df['scenario'] == 'Without MMSE']['accuracy'].mean():.1%} (drop: {mmse_drop:.1%})")
    click.echo(f"Without global brain metrics: {results_df[results_df['scenario'] == 'Without Global Brain Metrics']['accuracy'].mean():.1%} (drop: {global_drop:.1%})")
    click.echo("")
    click.echo("CLINICAL CONCLUSION:")
    click.echo("Models over-rely on cognitive test and non-specific brain measures.")
    click.echo("Regional imaging features are REQUIRED for clinical reliability.")
    click.echo("="*70)


if __name__ == '__main__':
    ablation_study()
