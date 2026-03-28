import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return {str(k): convert_to_serializable(v) for k, v in obj.to_dict().items()}
    elif isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def save_json(data, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    serializable_data = convert_to_serializable(data)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_data, f, indent=4)
    
    print(f"JSON saved to {filepath}")


def load_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def plot_feature_importance(feature_importance_df, model_type, output_path, top_n=20):
    plt.figure(figsize=(10, 8))
    
    top_features = feature_importance_df.head(top_n)
    
    if model_type in ['random_forest', 'gradient_boosting', 'xgboost', 'adaboost']:
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances - {model_type.replace("_", " ").title()}')
    elif model_type in ['logistic_regression', 'svm']:
        sns.barplot(data=top_features, x='abs_coefficient', y='feature', palette='viridis')
        plt.xlabel('Absolute Coefficient')
        plt.title(f'Top {top_n} Feature Coefficients - {model_type.replace("_", " ").title()}')
    
    plt.ylabel('Feature')
    plt.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Feature importance plot saved to {output_path}")


def plot_confusion_matrix(confusion_matrix, output_path):
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Dementia', 'Dementia'],
                yticklabels=['No Dementia', 'Dementia'])
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix plot saved to {output_path}")


def print_metrics(metrics, model_type):
    print(f"\n{'='*50}")
    print(f"{model_type.upper()} - Model Evaluation Metrics")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"{'='*50}\n")
