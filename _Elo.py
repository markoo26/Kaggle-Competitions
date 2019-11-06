import pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy.stats import skew
train_data = pd.read_csv('C:\\Users\\Marek.Pytka\\Desktop\\Inne szkolenia\\Elo\\train.csv')
test_data = pd.read_csv('C:\\Users\\Marek.Pytka\\Desktop\\Inne szkolenia\\Elo\\test.csv')
historical_transactions = pd.read_csv('C:\\Users\\Marek.Pytka\\Desktop\\Inne szkolenia\\Elo\\historical_transactions.csv')
new_historical_transaction = pd.read_csv('C:\\Users\\Marek.Pytka\\Desktop\\Inne szkolenia\\Elo\\new_merchant_transactions.csv')
merchants = pd.read_csv('C:\\Users\\Marek.Pytka\\Desktop\\Inne szkolenia\\Elo\\merchants.csv')

### One-card Data 
train_data['card_id'][0]
oc_data = historical_transactions[historical_transactions['card_id'] == train_data['card_id'][0]]
oc_data.columns

## transaction data:

# sum of money spent in last 3 months (standardized data)

tr_hist_tran = historical_transactions.iloc[0:20, :]
new_historical_transaction.columns
train_data.head()

##############################

    strings = [] 
    floats = [] 
    integers = []

    data_types = historical_transactions.dtypes
    
    for i in range(0,len(data_types[data_types == 'object'])):
        strings.append(data_types[data_types == 'object'].index[i])
        
    for i in range(0,len(data_types[data_types == 'int64'])):
        integers.append(data_types[data_types == 'int64'].index[i])
    
    for i in range(0,len(data_types[data_types == 'float64'])):
        floats.append(data_types[data_types == 'float64'].index[i])
    
    dummy_summary = []

    
    for i in range(0,len(strings)):
        
        print(type(np.count_nonzero(str(np.unique(historical_transactions[strings].iloc[:,i])))))
        print(type(str(historical_transactions[strings].columns[i])))
        dummy_summary.append([str(col_name), str(counter)])
    
    dataset[floats].hist(bins=30)
    plt.title('Histograms for float variables')
    
    print('Variables that are strings:')
    print(strings)
    print('Variables that are floats:')
    print(floats)
    print('Variables that are integers:')
    print(integers)

    print('Skewness calculation for float variables:')

    for i in range(0,len(floats)):
        colname = dataset[floats].columns[i]
        skewness = skew(dataset[colname])
        print(colname + ': ' + str(skewness))
        
initial_data_summary(historical_transactions)

#############################



def initial_data_summary(dataset):

    strings = [] 
    floats = [] 
    integers = []

    data_types = dataset.dtypes
    
    for i in range(0,len(data_types[data_types == 'object'])):
        strings.append(data_types[data_types == 'object'].index[i])
        
    for i in range(0,len(data_types[data_types == 'int64'])):
        integers.append(data_types[data_types == 'int64'].index[i])
    
    for i in range(0,len(data_types[data_types == 'float64'])):
        floats.append(data_types[data_types == 'float64'].index[i])
    
    dummy_summary = []
    
    for i in range(0,len(strings)):
        
        counter = np.count_nonzero(np.unique(dataset[strings].iloc[:,i]))
        col_name = dataset[strings].columns[i]
        dummy_summary.append([col_name, counter])
    
    dataset[floats].hist(bins=30)
    plt.title('Histograms for float variables')
    
    print('Variables that are strings:')
    print(strings)
    print('Variables that are floats:')
    print(floats)
    print('Variables that are integers:')
    print(integers)

    print('Skewness calculation for float variables:')

    for i in range(0,len(floats)):
        colname = dataset[floats].columns[i]
        skewness = skew(dataset[colname])
        print(colname + ': ' + str(skewness))
        
initial_data_summary(historical_transactions)


