import click
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


@click.command()
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
def evaluate_all(data_dir, model_dir, output_dir):
    click.echo("="*70)
    click.echo("OASIS - Evaluating All Models")
    click.echo("="*70)
    
    models_to_evaluate = [
        'knn',
        'xgboost',
        'gradient_boosting',
        'svm',
        'naive_bayes',
        'adaboost'
    ]
    
    click.echo(f"\nEvaluating {len(models_to_evaluate)} new models...")
    click.echo("(Random Forest and Logistic Regression already evaluated)")
    
    import subprocess
    
    for i, model in enumerate(models_to_evaluate, 1):
        click.echo(f"\n{'='*70}")
        click.echo(f"[{i}/{len(models_to_evaluate)}] Evaluating {model.upper()}")
        click.echo(f"{'='*70}")
        
        try:
            result = subprocess.run(
                [sys.executable, 'scripts/evaluate.py', '--model', model,
                 '--data-dir', data_dir, '--model-dir', model_dir,
                 '--output-dir', output_dir],
                capture_output=True,
                text=True,
                check=True
            )
            click.echo(result.stdout)
            click.echo(f"✓ {model} evaluation completed")
        except subprocess.CalledProcessError as e:
            click.echo(f"✗ Error evaluating {model}:")
            click.echo(e.stderr)
    
    click.echo(f"\n{'='*70}")
    click.echo("✓ All model evaluations completed!")
    click.echo(f"{'='*70}")
    click.echo(f"\nAll results saved to: {output_dir}/")
    click.echo(f"\nGenerated files for each model:")
    click.echo(f"  - [model]_confusion_matrix.png")
    click.echo(f"  - [model]_feature_importance.png (if available)")
    click.echo(f"  - [model]_evaluation_report.txt")


if __name__ == '__main__':
    evaluate_all()
