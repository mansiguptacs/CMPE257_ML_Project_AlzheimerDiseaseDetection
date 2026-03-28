import click
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.models import MLModel
from src.utils import load_json, plot_feature_importance, plot_confusion_matrix, print_metrics


@click.command()
@click.option('--model',
              type=click.Choice(['random_forest', 'logistic_regression', 'svm', 'xgboost', 'gradient_boosting', 'knn', 'naive_bayes', 'adaboost']),
              required=True,
              help='Model type to evaluate')
@click.option('--data-dir',
              type=click.Path(exists=True),
              default='data/processed/oasis1',
              help='Directory containing processed data')
@click.option('--model-dir',
              type=click.Path(exists=True),
              default='models/phase1_oasis1',
              help='Directory containing trained models')
@click.option('--output-dir',
              type=click.Path(),
              default='results/phase1_oasis1',
              help='Output directory for evaluation results')
def evaluate(model, data_dir, model_dir, output_dir):
    click.echo("="*60)
    click.echo(f"OASIS Model Evaluation Pipeline - {model.upper()}")
    click.echo("="*60)
    
    data_path = Path(data_dir)
    model_path = Path(model_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    click.echo(f"\n[1/6] Loading test data from: {data_dir}")
    X_test = pd.read_csv(data_path / 'X_test.csv')
    y_test = pd.read_csv(data_path / 'y_test.csv')['target']
    click.echo(f"  ✓ Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    click.echo(f"\n[2/6] Loading trained model from: {model_dir}")
    ml_model = MLModel(model_type=model)
    ml_model.load_model(model_path / f'{model}_model.pkl')
    ml_model.feature_names = X_test.columns.tolist()
    
    click.echo(f"\n[3/6] Evaluating model...")
    metrics = ml_model.evaluate(X_test, y_test)
    print_metrics(metrics, model)
    
    click.echo(f"\n[4/6] Generating feature importance visualization...")
    feature_importance = ml_model.get_feature_importance()
    importance_plot = None
    if feature_importance is not None:
        importance_plot = output_path / f'{model}_feature_importance.png'
        plot_feature_importance(feature_importance, model, importance_plot, top_n=20)
        
        click.echo(f"\n  Top 15 Important Features:")
        for idx, row in feature_importance.head(15).iterrows():
            if model in ['random_forest', 'gradient_boosting', 'xgboost', 'adaboost']:
                click.echo(f"    {idx+1}. {row['feature']}: {row['importance']:.4f}")
            else:
                click.echo(f"    {idx+1}. {row['feature']}: {row['coefficient']:.4f} (abs: {row['abs_coefficient']:.4f})")
    else:
        click.echo(f"  Note: {model} does not provide feature importance")
    
    click.echo(f"\n[5/6] Generating confusion matrix visualization...")
    cm_plot = output_path / f'{model}_confusion_matrix.png'
    plot_confusion_matrix(metrics['confusion_matrix'], cm_plot)
    
    click.echo(f"\n[6/6] Saving detailed evaluation report...")
    report_file = output_path / f'{model}_evaluation_report.txt'
    
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"{model.upper()} - Detailed Evaluation Report\n")
        f.write("="*70 + "\n\n")
        
        f.write("PERFORMANCE METRICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1 Score:  {metrics['f1_score']:.4f}\n")
        f.write(f"ROC AUC:   {metrics['roc_auc']:.4f}\n\n")
        
        f.write("CONFUSION MATRIX\n")
        f.write("-"*70 + "\n")
        cm = metrics['confusion_matrix']
        f.write(f"True Negatives:  {cm[0][0]}\n")
        f.write(f"False Positives: {cm[0][1]}\n")
        f.write(f"False Negatives: {cm[1][0]}\n")
        f.write(f"True Positives:  {cm[1][1]}\n\n")
        
        if feature_importance is not None:
            f.write("TOP 20 FEATURE IMPORTANCE\n")
            f.write("-"*70 + "\n")
            for idx, row in feature_importance.head(20).iterrows():
                if model in ['random_forest', 'gradient_boosting', 'xgboost', 'adaboost']:
                    f.write(f"{idx+1:2d}. {row['feature']:30s} {row['importance']:.6f}\n")
                else:
                    f.write(f"{idx+1:2d}. {row['feature']:30s} {row['coefficient']:+.6f}\n")
        else:
            f.write("FEATURE IMPORTANCE\n")
            f.write("-"*70 + "\n")
            f.write(f"{model} does not provide feature importance metrics.\n")
        
        f.write("\n" + "="*70 + "\n")
    
    click.echo(f"  ✓ Report saved to {report_file}")
    
    click.echo(f"\n{'='*60}")
    click.echo("✓ Model evaluation completed successfully!")
    click.echo(f"{'='*60}")
    click.echo(f"\nGenerated files:")
    if importance_plot:
        click.echo(f"  - {importance_plot}")
    click.echo(f"  - {cm_plot}")
    click.echo(f"  - {report_file}")


if __name__ == '__main__':
    evaluate()
