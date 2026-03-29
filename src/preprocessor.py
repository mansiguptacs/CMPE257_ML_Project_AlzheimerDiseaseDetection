import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import numpy as np

class OASISPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None

    def identify_target_and_features(self, df):
        potential_targets = ['CDR', 'Dementia', 'Group']
        target_col = None
        
        for col in potential_targets:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            raise ValueError(f"No target column found. Looking for: {potential_targets}")
        
        exclude_cols = [target_col, 'ID', 'Subject ID', 'Subject_ID', 'MRI ID', 'dataset', 'Hand']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        return target_col, feature_cols
    
    def handle_missing_values(self, df, strategy='median'):
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        print("Missing values before handling:")
        print(df_clean.isnull().sum())
        print(numeric_cols)
        print(categorical_cols)
        if strategy == 'median':
            for col in numeric_cols: 
                if df_clean[col].isnull().sum() > 0:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        elif strategy == 'mean':
            for col in numeric_cols:
                if df_clean[col].isnull().sum() > 0:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown')
        
        #remove tuple that ends with MR2 in ID
        df_clean = df_clean[~df_clean['ID'].str.endswith('MR2')]
        return df_clean
    
    def encode_categorical(self, data, categorical_cols):
        le = LabelEncoder()
        df_encoded = data.copy()
        for col in categorical_cols:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
        print("Finished Encoding features")
        return df_encoded
    
    def create_binary_target(self, df, target_col):
        df_binary = df.copy()
        if target_col == 'CDR':
            df_binary['target'] = (df_binary[target_col] > 0).astype(int)
        elif target_col in ['Dementia', 'Group']:
            if df_binary[target_col].dtype == 'object':
                df_binary['target'] = (df_binary[target_col] != 'Nondemented').astype(int)
            else:
                df_binary['target'] = df_binary[target_col]
        else:
            df_binary['target'] = df_binary[target_col]
        
        return df_binary

    def scale_features(self, X_train, X_test, binary_cols=None):
        """Scale only continuous numeric features; leave binary columns untouched."""
        if binary_cols is None:
            binary_cols = []
        
        cols_to_scale = [c for c in X_train.columns if c not in binary_cols]
        
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        if cols_to_scale:
            X_train_scaled[cols_to_scale] = self.scaler.fit_transform(X_train[cols_to_scale])
            X_test_scaled[cols_to_scale] = self.scaler.transform(X_test[cols_to_scale])
        
        return X_train_scaled, X_test_scaled
    
    def preprocess_pipeline(self, df, target_col, feature_cols, test_size=0.2, random_state=42, subject_level_split=True):
        df_clean = self.handle_missing_values(df)
        
        categorical_cols = df_clean[feature_cols].select_dtypes(include=['object']).columns.tolist()
        df_encoded = self.encode_categorical(df_clean, categorical_cols)
        
        df_final = self.create_binary_target(df_encoded, target_col)
        
        df_final = df_final.dropna(subset=['target'])
        
        X = df_final[feature_cols]
        y = df_final['target']
        
        X = X.select_dtypes(include=[np.number])
        
        # Drop columns that are entirely NaN (e.g., 'Delay' in cross-sectional data)
        nan_cols = X.columns[X.isnull().all()].tolist()
        if nan_cols:
            print(f"  Dropping columns with all NaN values: {nan_cols}")
            X = X.drop(columns=nan_cols)
        
        self.feature_names = X.columns.tolist()
        
        if subject_level_split and 'Subject_ID' in df_final.columns:
            print("Using subject-level split to prevent data leakage...")
            unique_subjects = df_final['Subject_ID'].unique()
            
            subject_targets = df_final.groupby('Subject_ID')['target'].first()
            
            train_subjects, test_subjects = train_test_split(
                unique_subjects, 
                test_size=test_size, 
                random_state=random_state,
                stratify=subject_targets
            )
            
            train_mask = df_final['Subject_ID'].isin(train_subjects)
            test_mask = df_final['Subject_ID'].isin(test_subjects)
            
            X_train = X[train_mask]
            X_test = X[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]
            
            print(f"  Split by subjects: {len(train_subjects)} train, {len(test_subjects)} test")
            print(f"  Total samples: {len(X_train)} train, {len(X_test)} test")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        
        # Identify binary/categorical columns that should not be scaled
        binary_cols = []
        for col in X_train.columns:
            unique_vals = X_train[col].dropna().unique()
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                binary_cols.append(col)
        
        if binary_cols:
            print(f"  Binary columns (not scaled): {binary_cols}")
        
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test, binary_cols=binary_cols)
        
        return X_train_scaled, X_test_scaled, y_train.reset_index(drop=True), y_test.reset_index(drop=True)
    
    def save_preprocessor(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.scaler, output_dir / 'scaler.pkl')
        joblib.dump(self.label_encoders, output_dir / 'label_encoders.pkl')
        joblib.dump(self.feature_names, output_dir / 'feature_names.pkl')
        
        print(f"Preprocessor saved to {output_dir}")
    
    def load_preprocessor(self, input_dir):
        input_dir = Path(input_dir)
        
        self.scaler = joblib.load(input_dir / 'scaler.pkl')
        self.label_encoders = joblib.load(input_dir / 'label_encoders.pkl')
        self.feature_names = joblib.load(input_dir / 'feature_names.pkl')
        
        print(f"Preprocessor loaded from {input_dir}")
    
def read_data():
    DATA_PATH = "./data/oasis_cross-sectional-5708aa0a98d82080.xlsx"

    data = pd.read_excel(DATA_PATH)

    print("Raw data shape:", data.shape)
    print("Columns:", data.columns.tolist())
    print(data.head())
    return data

