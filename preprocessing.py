import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

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
        
        print("Missing values after handling:", df_clean.isnull().sum())
        return df_clean
    
    def encode_categorical(self, data, categorical_cols):
        le = LabelEncoder()
        df_encoded = df.copy()
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

#Scale features
def scale_features(X_train, X_test):
    X_train_scaled = self.scaler.fit_transform(X_train)
    X_test_scaled = self.scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled

#Handling missing values
#Removing columns with all null values, same value and ID
#Removing data where target is null
def handle_missing_values( data):

    #Check for missing values
    print("Missing values in data:", data.isnull().sum())
    #missing vallues in %
    print("Missing values in data:", data.isnull().sum() / len(data) * 100)
    data['CDR'] = data['CDR'].fillna(0)
    data['MMSE'] = data['MMSE'].fillna(data['MMSE'].median())
    data['SES'] = data['SES'].fillna(data['SES'].median())
    data['Educ'] = data['Educ'].fillna(data['Educ'].median())
    data['Delay'] = data['Delay'].fillna(data['Delay'].median())

    #Removing unrelevant records with missing MRI Images means ID ending with MRI2
    data = data[~data['ID'].str.endswith('MR2')]
    return data

def scale_features(X_train, X_test):
    print("Started Scaling features")
    scaler = StandardScaler()

    num_cols = ['Age', 'Educ', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
    print("Finished Scaling features")
    return X_train_scaled, X_test_scaled

def encode_features(data):
    print("Started Encoding features")
    le = LabelEncoder()
    if 'M/F' in data.columns:
        data['Gender'] = le.fit_transform(data['M/F'])
        data.drop(columns=['M/F'], inplace=True)
    if 'Hand' in data.columns:
        data['Hand'] = le.fit_transform(data['Hand'])
    print(data['Gender'].head())
    #print(data['Hand'].head())
    print("Finished Encoding features")
    return data

def encode_target(data):
    print("Started Encoding target")
    #id CDR > 0 , then 1 else 0
    data['CDR'] = data['CDR'].apply(lambda x: 1 if x > 0 else 0)
    print("Finished Encoding target")
    return data

def save_data(X_train_scaled, X_test_scaled, y_train, y_test):
    print("X_train_scaled:", X_train_scaled.head())
    print("X_test_scaled:", X_test_scaled.head())
    print("y_train:", y_train.head())
    print("y_test:", y_test.head())
    X_train_scaled.to_csv("./data/X_train.csv", index=False)
    X_test_scaled.to_csv("./data/X_test.csv", index=False)
    y_train.to_csv("./data/y_train.csv", index=False)
    y_test.to_csv("./data/y_test.csv", index=False)

def perform_feature_selection(data):

    print("Dropping columns ID, Delay and Hand")
    new_data = data.drop(columns=['ID','Delay','Hand'])
    print("Shape of data after feature selection:", new_data.shape)
    return new_data

def perform_test_train_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
    return X_train, X_test, y_train, y_test

def correlation_matrix(X, y):
    print("Correlation matrix for cleaned data")
    corr = X.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.show()

def visualise_target_distribution(target):
    plt.figure(figsize=(5,3))
    #add percentage in the graph
    sns.countplot(x=target, palette=['green', 'red'])
    plt.title("Binary Target Distribution")
    plt.xlabel("Target")
    plt.ylabel("Count")
    plt.xticks(ticks=[0, 1], labels=['Healthy(0)', 'Dementia(1)'])
    plt.show()

def visualise_feature_distribution(df):
    features = ['Age', 'Educ', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
    healthy = df[df['CDR'] == 0]
    dementia = df[df['CDR'] == 1]

    # Create subplots grid
    fig, axes = plt.subplots(2, 4, figsize=(10, 6))
    axes = axes.flatten()
    for i, col in enumerate(features):
        ax = axes[i]

        # Healthy (green)
        sns.histplot(healthy[col], bins=20, color='green',
                    alpha=0.6, label='Healthy', ax=ax)

        # Dementia (red)
        sns.histplot(dementia[col], bins=20, color='red',
                    alpha=0.6, label='Dementia', ax=ax)

        ax.set_title(col)
        ax.legend()

    # Remove empty subplot if needed
    if len(features) < len(axes):
        fig.delaxes(axes[-1])

    plt.suptitle('Feature Distributions: Healthy vs Dementia')
    plt.tight_layout()
    plt.show()

def split_and_save_data(data):
    feature_cols = ['Gender', 'Age', 'Educ', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
    target_col = 'CDR'
    #Split data
    print("5. Splitting data into training and testing sets")
    X = data[feature_cols]
    y = data[target_col]
    X_train, X_test, y_train, y_test = perform_test_train_split(X, y)

    #Scale features
    print("6. Scaling features")
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    print("7. Saving data for future use")
    save_data(X_train_scaled, X_test_scaled, y_train, y_test)

    print("Class distribution (train): ", y_train.value_counts().to_dict())
    print("Class distribution (test): ", y_test.value_counts().to_dict())
    return X_train_scaled, X_test_scaled, y_train, y_test

def preprocess_oasis1(data=[]):
    if data.empty:
        print("1. Reading data from CSV file")
        data = read_data()
    print("Data loading completed")
    print("2. Handling missing values")
    new_data = handle_missing_values(data)
    print(new_data.describe())
    print(new_data.info())

    print("Shape of data after handling missing values:", new_data.shape)
    print("3. Dropping columns ID, Delay and Hand")
    new_data = perform_feature_selection(new_data)

    #Encode target
    print("4. Encoding \n Encoding target")
    new_data = encode_target(new_data)

    #Encode features
    print("Encoding features")
    new_data = encode_features(new_data)
    feature_cols = ['Gender', 'Age', 'Educ', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
    target_col = 'CDR'

    print("Class distribution for target(CDR): ", new_data[target_col].value_counts().to_dict())
    return new_data

if __name__ == '__main__':
    print("1. Reading data from CSV file")
    data = read_data()
    new_data = preprocess_oasis1(data)
    split_and_save_data(new_data)
    print("8. Correlation matrix")
    correlation_matrix(new_data, new_data['CDR'])
    print("9. Visualising target distribution")
    visualise_target_distribution(new_data['CDR'])
    print("10. Visualising feature distribution")
    visualise_feature_distribution(new_data)
