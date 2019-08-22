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

all_data_na = (sum(train_tran.isnull()) / len(train_tran))

train_tran.fillna(False).select_dtypes(include['bool']).sum(axis=1)