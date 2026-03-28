from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


class MLModel:
    def __init__(self, model_type='random_forest', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._initialize_model()
        self.feature_names = None
        
    def _initialize_model(self):
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'svm':
            return SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=self.random_state
            )
        elif self.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is not installed. Install it with: pip install xgboost")
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='logloss'
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state
            )
        elif self.model_type == 'knn':
            return KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                n_jobs=-1
            )
        elif self.model_type == 'naive_bayes':
            return GaussianNB()
        elif self.model_type == 'adaboost':
            return AdaBoostClassifier(
                n_estimators=100,
                learning_rate=1.0,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train):
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        
        self.model.fit(X_train, y_train)
        print(f"{self.model_type} model trained successfully")
        
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        return metrics
    
    def get_feature_importance(self):
        if self.model_type in ['random_forest', 'gradient_boosting', 'adaboost']:
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            feature_importance = pd.DataFrame({
                'feature': [self.feature_names[i] if self.feature_names else f'feature_{i}' for i in indices],
                'importance': importances[indices]
            })
            
            return feature_importance
        
        elif self.model_type == 'xgboost':
            if XGBOOST_AVAILABLE:
                importances = self.model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                feature_importance = pd.DataFrame({
                    'feature': [self.feature_names[i] if self.feature_names else f'feature_{i}' for i in indices],
                    'importance': importances[indices]
                })
                
                return feature_importance
            return None
        
        elif self.model_type == 'logistic_regression':
            coefficients = self.model.coef_[0]
            indices = np.argsort(np.abs(coefficients))[::-1]
            
            feature_importance = pd.DataFrame({
                'feature': [self.feature_names[i] if self.feature_names else f'feature_{i}' for i in indices],
                'coefficient': coefficients[indices],
                'abs_coefficient': np.abs(coefficients[indices])
            })
            
            return feature_importance
        
        elif self.model_type == 'svm':
            if hasattr(self.model, 'coef_') and self.model.kernel == 'linear':
                coefficients = self.model.coef_[0]
                indices = np.argsort(np.abs(coefficients))[::-1]
                
                feature_importance = pd.DataFrame({
                    'feature': [self.feature_names[i] if self.feature_names else f'feature_{i}' for i in indices],
                    'coefficient': coefficients[indices],
                    'abs_coefficient': np.abs(coefficients[indices])
                })
                
                return feature_importance
            return None
        
        else:
            return None
    
    def save_model(self, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, output_path)
        print(f"Model saved to {output_path}")
    
    def load_model(self, input_path):
        self.model = joblib.load(input_path)
        print(f"Model loaded from {input_path}")
