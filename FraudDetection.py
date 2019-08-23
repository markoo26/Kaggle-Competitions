import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import os, time
from scipy import stats

os.chdir("C:\\Users\\Marek\\Desktop\\Python\\Kaggle\\FraudDetection")

start = time.time()
train_tran = pd.read_csv("train_transaction.csv")
test_tran = pd.read_csv("test_transaction.csv")
train_id = pd.read_csv("train_identity.csv")
test_id = pd.read_csv("test_identity.csv")
end = time.time()

y_train = train_tran["isFraud"]
train_tran = train_tran.drop(columns ="isFraud")

print(end-start) ## Execution time - 87 seconds

def pd_array_columns(pd_array):
    for i in range(1, len(pd_array.columns) + 1):
        description = str(i-1) + ':' + str(pd_array.columns[i-1])
        print(description)
        

pd_array_columns(train_tran)

### Outliers ###

fig,ax = plt.subplots()
ax.scatter(train_tran["TransactionAmt"], train_tran["card1"])
plt.ylabel("Transaction Amount")
plt.xlabel("Card1 Amount")
plt.show()

train_tran["TransactionAmt"] = np.log1p(train_tran["TransactionAmt"])

sns.distplot(train_tran["TransactionAmt"])

all_data_na = train_tran.isnull().sum() / len(train_tran) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending = True)[:30]

all_data_nv = train_tran[normal_variables].isnull().sum() / train_tran.shape[0] * 100

f, ax = plt.subplots(figsize=(15,12))
plt.xticks(rotation = 90)
sns.barplot(all_data_nv.index, all_data_nv)
plt.xlabel('Variables')
plt.ylabel('Percentage of missing values')
plt.title('Potentially best variables')

normal_variables = []

    for i in range(1, 17):
        normal_variables.append(train_tran.columns[i-1])

train_tran_nv = train_tran[normal_variables] ### NULL values need to be handled
corrmat = train_tran_nv.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(train_tran_nv, square = True, annot = True)

corrmat.pivot_table()
