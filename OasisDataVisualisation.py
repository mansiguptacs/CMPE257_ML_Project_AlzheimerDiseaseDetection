import pandas as pd
from preprocessing import preprocess_oasis1
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "./data/oasis_cross-sectional-5708aa0a98d82080.xlsx"

# We let pandas infer types; for this dataset it usually does the right thing.
data = pd.read_excel(DATA_PATH)

print("Raw data shape:", data.shape)
print("Columns:", data.columns.tolist())
print(data.head())


def feature_distribution(df):
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

def correlation_matrix(X, y):
    print("Correlation matrix for cleaned data")
    corr = X.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.show()

def target_distribution(target):
    plt.figure(figsize=(5,3))
    #add percentage in the graph
    sns.countplot(x=target, palette=['green', 'red'])
    plt.title("Binary Target Distribution")
    plt.xlabel("Target")
    plt.ylabel("Count")
    plt.xticks(ticks=[0, 1], labels=['Healthy(0)', 'Dementia(1)'])
    plt.show()
print("Enter 1 for feature Distibution\n 2. Target Distribution \n 3. Correlation Matrix")
choice = input("Enter your choice: ")
data = preprocess_oasis1(data)
if choice == "1":
    print("Visualising data")
    feature_distribution(data)
elif choice == "2":
    print("Target Distribution of data")
    target_distribution(data['CDR'])
elif choice == "3":
    print("Visualising Correlation matrix of data")
    correlation_matrix(data, data['CDR'])
    

    

    

