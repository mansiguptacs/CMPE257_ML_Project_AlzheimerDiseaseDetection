import click
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.preprocessor import OASISPreprocessor
from src.utils import save_json


@click.command()
@click.option('--input',
              type=click.Path(exists=True),
              default='oasis_cross-sectional-5708aa0a98d82080.xlsx',
              help='Path to OASIS-1 cross-sectional dataset')
@click.option('--output-dir',
              type=click.Path(),
              default='data/processed/oasis1',
              help='Output directory for processed data')
@click.option('--test-size',
              type=float,
              default=0.2,
              help='Test set size (default: 0.2)')
@click.option('--random-state',
              type=int,
              default=42,
              help='Random state for reproducibility')
def preprocess_oasis1(input, output_dir, test_size, random_state):
    click.echo("="*60)
    click.echo("OASIS-1 (Cross-Sectional) Data Preprocessing")
    click.echo("="*60)
    
    click.echo(f"\nLoading OASIS-1 dataset from: {input}")
    df = pd.read_excel(input)
    df['dataset'] = 'cross_sectional'
    if 'ID' in df.columns:
        df['Subject_ID'] = df['ID']
    click.echo(f"Loaded {df.shape[0]} samples with {df.shape[1]} features")
    
    preprocessor = OASISPreprocessor()
    
    click.echo(f"\nIdentifying target and features...")
    target_col, feature_cols = preprocessor.identify_target_and_features(df)
    click.echo(f"Target column: {target_col}")
    click.echo(f"Feature columns: {len(feature_cols)}")
    click.echo(f"Features: {', '.join(feature_cols[:5])}...")
    
    click.echo(f"\nAnalyzing missing values...")
    missing_before = df.isnull().sum().sum()
    click.echo(f"Total missing values: {missing_before}")
    
    click.echo(f"\nRunning preprocessing pipeline...")
    click.echo(f"Handling missing values")
    click.echo(f"Encoding categorical variables")
    click.echo(f"Creating binary target")
    click.echo(f"Splitting data (test_size={test_size}, random split)")
    click.echo(f"Scaling features")
    
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
        df, target_col, feature_cols, test_size=test_size, random_state=random_state, subject_level_split=False
    )
    
    click.echo(f"\nPreprocessing results:")
    click.echo(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    click.echo(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    click.echo(f"Class distribution (train): {y_train.value_counts().to_dict()}")
    click.echo(f"Class distribution (test): {y_test.value_counts().to_dict()}")
    
    click.echo(f"\nSaving processed data to: {output_dir}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    X_train.to_csv(output_path / 'X_train.csv', index=False)
    X_test.to_csv(output_path / 'X_test.csv', index=False)
    y_train.to_csv(output_path / 'y_train.csv', index=False, header=['target'])
    y_test.to_csv(output_path / 'y_test.csv', index=False, header=['target'])
    
    click.echo(f"X_train.csv")
    click.echo(f"X_test.csv")
    click.echo(f"y_train.csv")
    click.echo(f"y_test.csv")
    
    click.echo(f"\nSaving preprocessing results...")
    preprocessor.save_preprocessor(output_path)
    
    metadata = {
        'dataset': 'OASIS-1 (Cross-Sectional)',
        'target_column': target_col,
        'feature_columns': feature_cols,
        'final_features': preprocessor.feature_names,
        'train_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'num_features': int(X_train.shape[1]),
        'test_size': test_size,
        'random_state': random_state,
        'subject_level_split': False,
        'class_distribution_train': y_train.value_counts().to_dict(),
        'class_distribution_test': y_test.value_counts().to_dict()
    }
    save_json(metadata, output_path / 'preprocessing_metadata.json')
    
    click.echo(f"\n{'='*60}")
    click.echo("OASIS-1 preprocessing completed successfully!")
    click.echo(f"{'='*60}")
    click.echo(f"\nNext step: Train models on OASIS-1")
    click.echo(f"python scripts/train_all_models.py --data-dir data/processed_oasis1 --output-dir models_oasis1")


if __name__ == '__main__':
    preprocess_oasis1()
