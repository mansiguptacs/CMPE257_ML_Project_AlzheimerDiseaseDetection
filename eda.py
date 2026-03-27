import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "./data/oasis_cross-sectional-5708aa0a98d82080.xlsx"

data = pd.read_excel(DATA_PATH)

print("Raw data shape:", data.shape)
print("Columns:", data.columns.tolist())
print(data.head())
target = data['CDR']
data = data.drop(columns=['CDR'])


print("high-level overview")
print(data.info())
print(data.describe())

print("missing values are presents in following attributes: ")
#print(data.isnull().sum())

missing_fields = data.isnull().sum()>0
for fields in missing_fields.index:
    if missing_fields[fields] == True:
        print(fields)

print("Checking null/missing values in target field")
print(target.isnull().sum())

#sns.heatmap(data.isnull(), cbar=False)
#plt.show()

num_cols = ['Age', 'Educ', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']

for col in num_cols:
    plt.figure(figsize=(5,3))
    
    if data[col].dtype in ['int64', 'float64']:
        sns.histplot(data[col], kde=True)
    else:
        sns.countplot(x=data[col])
    
    plt.title(col)
    plt.xticks(rotation=45)
    plt.show()

corr = data.corr(numeric_only=True)

sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()
#add loop to use 3 features at a time
'''
for i in range(0, len(num_cols)):
    for j in range(i+1, len(num_cols)):
        num_cols_1 = num_cols[i]
        num_cols_2 = num_cols[j]

        #plt.figure(figsize=(10,10))
        #sns.pairplot(data[num_cols])
        plotting = sns.pairplot(data[[num_cols_1,num_cols_2]])
        plotting.fig.suptitle("Pairplot of Numerical Features", y=1.02)
        plt.show()
'''
""" 
num_cols_2 = num_cols[3:]

data[num_cols_2].hist(figsize=(12,8))
plt.show()
plt.figure(figsize=(10,10))
plotting = sns.pairplot(data[num_cols_2])
plotting.fig.suptitle("Pairplot of Numerical Features part 2", y=1.02)
plt.show() """

print("value counts for M/F")
print(data['M/F'].value_counts())
print("value counts for Hand")
print(data['Hand'].value_counts())

