### Import of the libraries

import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import os, time, webbrowser

from scipy import stats
from scipy.stats import norm, skew, poisson
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

### Set working directory

os.chdir("C:\\Users\\Marek\\Desktop\\Python\\Kaggle\\FraudDetection")

### Related competition

webbrowser.open_new_tab("https://www.kaggle.com/c/ieee-fraud-detection/data")

### Functions

def pd_array_columns(pd_array):
    for i in range(1, len(pd_array.columns) + 1):
        description = str(i-1) + ':' + str(pd_array.columns[i-1])
        print(description)

### Data import & preparation

start = time.time()
#raw_train =  pd.read_csv("train_transaction.csv")
train_tran = pd.read_csv("train_transaction.csv")
test_tran = pd.read_csv("test_transaction.csv")
train_id = pd.read_csv("train_identity.csv")
test_id = pd.read_csv("test_identity.csv")

full_train_data = train_tran.merge(train_id, on="TransactionID") #!#!# Znaczne ograniczenie zbioru danych przez inner joina
full_test_data = test_tran.merge(test_id, on="TransactionID")

### Split into training and test set 

X_train, X_test, y_train, y_test = train_test_split(full_train_data.drop("isFraud",1),
                                                    full_train_data["isFraud"],
                                                    test_size = .2, random_state = 0)

av_columns = list(full_train_data.columns.values)

end = time.time()
print(end-start) ## Execution time - 87 seconds        

### All columns & their indexes

pd_array_columns(X_train)

### Assignment to 'normal variables' (all with meaningful variable name)

mngful_variables = []

for i in range(1, 17):
    mngful_variables.append(X_train.columns[i-1])

for i in range(431,433):
    mngful_variables.append(X_train.columns[i])

### Identifying and standardizing numeric (int/float64) features

train_numeric = X_train.dtypes.isin([np.float64,np.int64])

train_numeric = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_indexes = [int(X_train.columns.get_loc(c)) for c in train_numeric if c in X_train.columns]


sc_X = StandardScaler()
sc_X.fit_transform(X_train[:,numeric_indexes])

]
### Outliers ###

fig,ax = plt.subplots()
ax.scatter(x= X_train["ProductCD"], y= y_train)
plt.ylabel("Transaction Amount")
plt.xlabel("Card1 Amount")
plt.show()

### Outliers 

### Log transformation of TransactionAmt #!#!# Check if all variables need that, loop with skewed vars

train_tran["TransactionAmt"] = np.log1p(train_tran["TransactionAmt"])
sns.distplot(train_tran["TransactionAmt"])

### Summary of NULL values

all_data_na = X_train.isnull().sum() / len(X_train) * 100
all_data_no_null = all_data_na.drop(all_data_na[all_data_na != 0].index).sort_values(ascending = True)[:30]
all_data_nv = X_train[numeric_indexes].isnull().sum() / train_tran.shape[0] * 100

X_train[mngful_variables]

f, ax = plt.subplots(figsize=(15,12))
plt.xticks(rotation = 90)
sns.barplot(all_data_nv.index, all_data_nv)
plt.xlabel('Variables')
plt.ylabel('Percentage of missing values')
plt.title('Potentially best variables')

### Summary of categorical variables

pc_summary = X_train.groupby(by = "ProductCD").count().iloc[:,1]
ed_summary = X_train.groupby(by = "P_emaildomain").count().iloc[:,1]


train_tran_nv = train_tran[normal_variables] ### NULL values need to be handled
corrmat = train_tran_nv.corr()

#plt.subplots(figsize=(12,9))
#sns.heatmap(train_tran_nv, square = True, annot = True)
#corrmat.pivot_table()

### Correlation between one variable and train #!#!# Czy tu w ogole jest sens liczyc korelacje ze zmienna 01?


raw_train["isFraud","TransactionAmt"].corr()
sc_X.fit_transform(test_tran)

train_notnull = train_stan.isnull().sum()==0
notnull_columns = train_notnull[train_notnull == True].index
notnull_columns = notnull_columns.tolist()

classifier = LogisticRegression()
classifier.fit(train_stan, y_train)

test_notnull = test_tran.isnull().sum()

y_pred = classifier.predict(train_stan)

for column in test_tran.columns:
    test_tran[column] = test_tran[column].fillna(1)

cm = confusion_matrix(y_train, y_pred)

sensitivity = cm[0,0]/(cm[0,0] + cm[1,0])
print("Sensitivity: " + str(sensitivity))
specificity = cm[1,1]/(cm[1,1] + cm[0,1])
print("Specificity: " + str(specificity))

fpr,tpr, thresholds = roc_curve(y_train, y_pred)
fig,ax = plt.subplots()
ax.plot(fpr,tpr)
ax.plot([0,1],[0,1], transform = ax.transAxes, ls = "--", c = ".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0,1.0])
plt.rcParams['font.size']=12
plt.title("ROC Curve for Fraud Classifier")
plt.xlabel("False Positive Rate (1-Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.grid(True)

auc(fpr,tpr)