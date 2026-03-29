import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.preprocessor import OASISPreprocessor

DATA_PATH = "./data/oasis_cross-sectional-5708aa0a98d82080.xlsx"

# We let pandas infer types; for this dataset it usually does the right thing.
data = pd.read_excel(DATA_PATH)

print("Raw data shape:", data.shape)
print("Columns:", data.columns.tolist())
print(data.head())

def perform_test_train_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
    return X_train, X_test, y_train, y_test

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

def preprocess_oasis1(data=[]):
    if data.empty:
        data = pd.read_excel(DATA_PATH)
    print("Data loading completed")
    print("2. Handling missing values")
    preprocessor = OASISPreprocessor()
    new_data = preprocessor.handle_missing_values(data)
    print(new_data.describe())

    print("Shape of data after handling missing values:", new_data.shape)
    print("3. Dropping columns ID, Delay and Hand")
    new_data = new_data.drop(columns=['ID','Delay','Hand'])

    #Encode target
    print("4. Encoding \n Encoding target")
    new_data = preprocessor.create_binary_target(new_data, 'CDR')

    #Encode features
    print("Encoding features")
    new_data = preprocessor.encode_categorical(new_data, ['M/F', 'Hand'])
    feature_cols = ['M/F', 'Age', 'Educ', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
    target_col = 'target'

    print(new_data.describe())
    print("Scaling features")
    X_train, X_test, y_train, y_test = perform_test_train_split(new_data[feature_cols], new_data[target_col])
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    print("Class distribution for target(CDR): ", new_data[target_col].value_counts().to_dict())
    #return pd.concat([X_train_scaled,X_test_scaled], axis=0), pd.concat([y_train,y_test], axis=0)
    return new_data

def feature_distribution(df=None):
    if df is None:
        df = pd.read_excel("./data/oasis_cross-sectional-5708aa0a98d82080.xlsx")
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

def correlation_matrix(df=None):
    if df is None:
        df = pd.read_excel("./data/oasis_cross-sectional-5708aa0a98d82080.xlsx")
    features = ['Age', 'Educ', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF','target']
    print("Correlation matrix for cleaned data")

    new_data = preprocess_oasis1(df)
    corr = new_data[features].corr()

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
    corr,
    mask=mask,
    annot=True,          # show correlation values
    fmt=".2f",           # 2 decimal places
    cmap="coolwarm",     # blue to red
    vmin=-1, vmax=1,     # correlation scale
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8})

    plt.title("Feature Correlation Matrix", fontsize=14, fontweight='bold')
    plt.xticks(rotation=0)
    plt.yticks(rotation=90)
    plt.tight_layout()
    plt.show()

def target_distribution(target=None):
    if(target is None):
        target = pd.read_excel("./data/oasis_cross-sectional-5708aa0a98d82080.xlsx")['CDR']
    plt.figure(figsize=(5,3))
    #add percentage in the graph
    target = [1 if x > 0 else 0 for x in target]
    sns.countplot(x=target, palette=['green', 'red'])
    plt.title("Binary Target Distribution")
    plt.xlabel("Target")
    plt.ylabel("Count")
    plt.xticks(ticks=[0, 1], labels=['Healthy(0)', 'Dementia(1)'])
    plt.show()
print("Enter 1 for feature Distibution\n 2. Target Distribution \n 3. Correlation Matrix")
choice = input("Enter your choice: ")
#data = preprocess_oasis1(data)
if choice == "1":
    print("Visualising data")
    feature_distribution()
elif choice == "2":
    print("Target Distribution of data")
    target_distribution(data['CDR'])
elif choice == "3":
    print("Visualising Correlation matrix of data")
    correlation_matrix()
    

    

    

