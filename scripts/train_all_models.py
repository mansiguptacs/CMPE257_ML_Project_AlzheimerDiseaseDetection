import click
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time
from sklearn.model_selection import StratifiedKFold, cross_validate

sys.path.append(str(Path(__file__).parent.parent))

from src.models import MLModel
from src.utils import save_json, print_metrics


@click.command()
@click.option('--data-dir',
              type=click.Path(exists=True),
              default='data/processed/oasis1',
              help='Directory containing processed data')
@click.option('--output-dir',
              type=click.Path(),
              default='models/phase1_oasis1',
              help='Output directory for trained models')
@click.option('--random-state',
              type=int,
              default=42,
              help='Random state for reproducibility')
@click.option('--cv-folds',
              type=int,
              default=5,
              help='Number of cross-validation folds')
def train_all(data_dir, output_dir, random_state, cv_folds):
    click.echo("="*70)
    click.echo("OASIS - Training All Models")
    click.echo("="*70)
    
    models_to_train = [
        'random_forest',
        'logistic_regression',
        'svm',
        'xgboost',
        'gradient_boosting',
        'knn',
        'naive_bayes',
        'adaboost'
    ]
    
    data_path = Path(data_dir)
    
    click.echo(f"\n[1/3] Loading processed data from: {data_dir}")
    X_train = pd.read_csv(data_path / 'X_train.csv')
    X_test = pd.read_csv(data_path / 'X_test.csv')
    y_train = pd.read_csv(data_path / 'y_train.csv')['target']
    y_test = pd.read_csv(data_path / 'y_test.csv')['target']
    
    click.echo(f"  ✓ Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    click.echo(f"  ✓ Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    click.echo(f"  ✓ Features: {list(X_train.columns)}")
    
    # Set up k-fold cross-validation
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    click.echo(f"\n[2/3] Training {len(models_to_train)} models with {cv_folds}-fold CV...")
    
    results = []
    
    for i, model_type in enumerate(models_to_train, 1):
        click.echo(f"\n{'='*70}")
        click.echo(f"Model {i}/{len(models_to_train)}: {model_type.upper()}")
        click.echo(f"{'='*70}")
        
        try:
            start_time = time.time()
            
            click.echo(f"  Initializing {model_type}...")
            ml_model = MLModel(model_type=model_type, random_state=random_state)
            
            # --- K-Fold Cross-Validation on training set ---
            click.echo(f"  Running {cv_folds}-fold cross-validation...")
            cv_results = cross_validate(
                ml_model.model, X_train, y_train,
                cv=skf, scoring=scoring, return_train_score=False, n_jobs=-1
            )
            
            cv_accuracy_mean = cv_results['test_accuracy'].mean()
            cv_accuracy_std = cv_results['test_accuracy'].std()
            cv_f1_mean = cv_results['test_f1'].mean()
            cv_f1_std = cv_results['test_f1'].std()
            cv_roc_auc_mean = cv_results['test_roc_auc'].mean()
            cv_roc_auc_std = cv_results['test_roc_auc'].std()
            cv_precision_mean = cv_results['test_precision'].mean()
            cv_recall_mean = cv_results['test_recall'].mean()
            
            click.echo(f"  CV Accuracy:  {cv_accuracy_mean:.4f} ± {cv_accuracy_std:.4f}")
            click.echo(f"  CV F1 Score:  {cv_f1_mean:.4f} ± {cv_f1_std:.4f}")
            click.echo(f"  CV ROC AUC:   {cv_roc_auc_mean:.4f} ± {cv_roc_auc_std:.4f}")
            
            # --- Train on full training set, evaluate on holdout test ---
            click.echo(f"  Training on full training set...")
            ml_model.train(X_train, y_train)
            
            click.echo(f"  Evaluating on holdout test set...")
            metrics = ml_model.evaluate(X_test, y_test)
            
            training_time = time.time() - start_time
            metrics['training_time'] = training_time
            
            # Store CV results in metrics
            metrics['cv_accuracy_mean'] = cv_accuracy_mean
            metrics['cv_accuracy_std'] = cv_accuracy_std
            metrics['cv_f1_mean'] = cv_f1_mean
            metrics['cv_f1_std'] = cv_f1_std
            metrics['cv_roc_auc_mean'] = cv_roc_auc_mean
            metrics['cv_roc_auc_std'] = cv_roc_auc_std
            metrics['cv_folds'] = cv_folds
            metrics['cv_fold_accuracies'] = cv_results['test_accuracy'].tolist()
            
            print_metrics(metrics, model_type)
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            model_file = output_path / f'{model_type}_model.pkl'
            ml_model.save_model(model_file)
            
            metrics_file = output_path / f'{model_type}_metrics.json'
            save_json(metrics, metrics_file)
            
            feature_importance = ml_model.get_feature_importance()
            if feature_importance is not None:
                importance_file = output_path / f'{model_type}_feature_importance.csv'
                feature_importance.to_csv(importance_file, index=False)
                click.echo(f"  ✓ Feature importance saved")
            
            results.append({
                'model': model_type,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'roc_auc': metrics['roc_auc'],
                'cv_accuracy_mean': cv_accuracy_mean,
                'cv_accuracy_std': cv_accuracy_std,
                'cv_f1_mean': cv_f1_mean,
                'cv_f1_std': cv_f1_std,
                'cv_roc_auc_mean': cv_roc_auc_mean,
                'cv_roc_auc_std': cv_roc_auc_std,
                'training_time': training_time
            })
            
            click.echo(f"  ✓ {model_type} completed in {training_time:.2f}s")
            
        except Exception as e:
            click.echo(f"  ✗ Error training {model_type}: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append({
                'model': model_type,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'roc_auc': 0.0,
                'cv_accuracy_mean': 0.0,
                'cv_accuracy_std': 0.0,
                'cv_f1_mean': 0.0,
                'cv_f1_std': 0.0,
                'cv_roc_auc_mean': 0.0,
                'cv_roc_auc_std': 0.0,
                'training_time': 0.0,
                'error': str(e)
            })
    
    click.echo(f"\n[3/3] Generating comparison report...")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('accuracy', ascending=False)
    
    comparison_file = Path(output_dir) / 'model_comparison.csv'
    results_df.to_csv(comparison_file, index=False)
    
    click.echo(f"\n{'='*70}")
    click.echo("MODEL COMPARISON - Ranked by Accuracy")
    click.echo(f"{'='*70}")
    click.echo(f"\n{'Model':<22} {'Test Acc':<10} {'CV Acc (mean±std)':<22} {'Test F1':<10} {'CV AUC':<10} {'Time(s)':<10}")
    click.echo("-"*84)
    
    for _, row in results_df.iterrows():
        cv_str = f"{row['cv_accuracy_mean']:.4f}±{row['cv_accuracy_std']:.4f}"
        click.echo(f"{row['model']:<22} {row['accuracy']:<10.4f} {cv_str:<22} {row['f1_score']:<10.4f} {row['cv_roc_auc_mean']:<10.4f} {row['training_time']:<10.2f}")
    
    click.echo(f"\n{'='*70}")
    click.echo("✓ All models trained successfully!")
    click.echo(f"{'='*70}")
    click.echo(f"\nComparison saved to: {comparison_file}")
    click.echo(f"\nBest model: {results_df.iloc[0]['model']} (Accuracy: {results_df.iloc[0]['accuracy']:.4f})")
    click.echo(f"\nTo evaluate individual models:")
    click.echo(f"  python scripts/evaluate.py --model [model_name]")


if __name__ == '__main__':
    train_all()
