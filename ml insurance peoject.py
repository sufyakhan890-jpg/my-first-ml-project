import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)  
pd.set_option('display.width', 1000)  
deta = pd.read_csv("insurance.csv")
#print(deta)
#print(deta.head)
#print(deta.info())
#print(deta.describe())
#print(deta.isnull().sum())
#print(deta.columns)
numaric_colume =['age', 'weight', 'height', 'income_lpa']
#for clo in numaric_colume:
   # plt.figure(figsize=[6,4])
    #sns.histplot(deta[clo],kde=True,bins= 20)
    #plt.show()
#sns.countplot(x=deta['insurance_premium_category'])
#plt.show() 
#plt.figure(figsize=[10,8])
#sns.countplot(x=deta['occupation'])
#plt.show()  
#plt.figure(figsize=[6,8])
#sns.heatmap(deta.corr(numeric_only =True),annot=True)
#plt.show()
df_cleaned = deta.copy()
#print(df_cleaned.head())
#value1 =df_cleaned['insurance_premium_category'].value_counts()
#print(value1)
#print(df_cleaned.dtypes)
#value2 =df_cleaned['smoker'].value_counts()
#print(value2)
df_cleaned['smoker'] = df_cleaned['smoker'].map({True: 1 , False: 0})
#print(df_cleaned.head())
#print(df_cleaned.dtypes)
#print(df_cleaned['city'].value_counts())
#print(df_cleaned['occupation'].value_counts())
#print(df_cleaned['insurance_premium_category'].value_counts())
df_cleaned = pd.get_dummies(
    df_cleaned, columns=['insurance_premium_category']
)
#print(df_cleaned.head())
#print(df_cleaned.dtypes)
df_cleaned = df_cleaned.drop(columns=["city"])
df_cleaned['insurance_premium_category_High'] = df_cleaned['insurance_premium_category_High'].map({True: 1 , False: 0})
df_cleaned['insurance_premium_category_Low'] = df_cleaned['insurance_premium_category_Low'].map({True: 1 , False: 0})
df_cleaned['insurance_premium_category_Medium'] = df_cleaned['insurance_premium_category_Medium'].map({True: 1 , False: 0})
df_cleaned = df_cleaned.rename(
    columns={
        'insurance_premium_category_High': 'platinum_package',
        'insurance_premium_category_Medium': 'gold_package',
        'insurance_premium_category_Low': 'silver_package',
    }
)
occupation_mapping = {
    'government_job': 'salaried',
    'private_job': 'salaried',
    'business_owner': 'business',
    'freelancer': 'business',
    'student': 'other',
    'retired': 'other',
    'unemployed': 'other',
}

df_cleaned['occupation'] = df_cleaned['occupation'].map(occupation_mapping)

df_cleaned = pd.get_dummies(
    df_cleaned, columns=['occupation'], dtype=int, drop_first=True
)

df_cleaned = df_cleaned.rename(
    columns={
        'occupation_salaried': 'occ_salaried',
        'occupation_business': 'occ_business',
        'occupation_other': 'occ_other',
    }
)

#print(df_cleaned.head())
from sklearn.preprocessing import StandardScaler

scale_cols = ['age', 'weight', 'height', 'income_lpa']

scaler = StandardScaler()

df_cleaned[scale_cols] = scaler.fit_transform(df_cleaned[scale_cols])
#print("before awplaing EDA")
#print(deta.head())
#print("after awpling EDA")
#print(df_cleaned.head())
from sklearn.model_selection import train_test_split

X = df_cleaned.drop(
    columns=[
        'platinum_package',
        'gold_package',
        'silver_package'
    ]
)

y = df_cleaned[
    [
        'platinum_package',
        'gold_package',
        'silver_package'
    ]
]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scale_cols = ['age', 'weight', 'height', 'income_lpa']

scaler = StandardScaler()

X_train.loc[:, scale_cols] = scaler.fit_transform(
    X_train[scale_cols]
)

X_test.loc[:, scale_cols] = scaler.transform(
    X_test[scale_cols]
)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train , y_train)
y_pred = model.predict(X_test)
#print(y_pred)
#print(y_test)
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
#print(y_pred)
y_pred_rounded = np.round(y_pred)
print('Rounded Predictions:\n', y_pred_rounded)
print('Actual y_test:\n', y_test.values)
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print('R2 Score:', r2)

n = X_test.shape[0]
p = X_test.shape[1]
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
print('Adjusted R2 Score:', adjusted_r2)