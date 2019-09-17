#### TO DO LIST ####

### Train_test split for the neural network to improve calculation time
### Add Quarter Index to the input month 

#####################

#### LEGEND

###!### Need to be finished

#### IMPORT OF THE LIBRARIES ####

import pandas as pd, numpy as np, matplotlib.pyplot as plt
import webbrowser, datetime
from pandas.plotting import autocorrelation_plot
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

import keras
from keras.models import Sequential ### do inicjalizacji ANN
from keras.layers import Dense ### do tworzenia layerow

import os
os.environ["PATH"] += os.pathsep + 'E:/Program Files (x86)/Graphviz2.38/bin/'

#### LINKS

#Kaggle Competition
webbrowser.open_new_tab('https://www.kaggle.com/marcoracer/predict-future-sales-with-randomforest/notebook')

#Kernel of MarcoRacer
webbrowser.open_new_tab('https://www.kaggle.com/marcoracer/predict-future-sales-with-randomforest/notebook')

# Setting Appropriate Execution Time Format
webbrowser.open_new_tab('http://strftime.org/')

# Deep Neural Networks for Regression Problems
webbrowser.open_new_tab('https://towardsdatascience.com/deep-neural-networks-for-regression-problems-81321897ca33')

# Reading about ANN from ML course
webbrowser.open_new_tab('http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf')

# Visualizing the ANN
webbrowser.open_new_tab('https://towardsdatascience.com/visualizing-artificial-neural-networks-anns-with-just-one-line-of-code-b4233607209e')

# Available activation functions for the ANN
webbrowser.open_new_tab('https://keras.io/activations/')

# Different cost functions
webbrowser.open_new_tab('https://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications')

# Stochastic Gradient Descent REading
webbrowser.open_new_tab('https://iamtrask.github.io/2015/07/27/python-network-part2/')

#### SCRIPT 

### Data import

train_data = pd.read_csv('C:\\Users\\Marek\\Desktop\\Python\\Kaggle\\FutureSales\\sales_train_v2.csv')
test_data = pd.read_csv('C:\\Users\\Marek\\Desktop\\Python\\Kaggle\\FutureSales\\test.csv')
items_categories = pd.read_csv('C:\\Users\\Marek\\Desktop\\Python\\Kaggle\\FutureSales\\item_categories.csv')
items = pd.read_csv('C:\\Users\\Marek\\Desktop\\Python\\Kaggle\\FutureSales\\items.csv')

### Join items and item categories for better overview

items = items.merge(items_categories, left_on='item_category_id', right_on='item_category_id', how='left')
train_data = train_data.merge(items, left_on ='item_id', right_on = 'item_id', how = 'left' )
train_data = train_data.sort_values(by = ['shop_id', 'item_id', 'date'])

### Assignment of quarter to the date_block_num variable

unique_blocks = np.unique(train_data['date_block_num'])

MQ_Dict = {}

for year in range(2013,2016):
    print(year)
    for quarter in range(1,5):
        print(quarter)
        var_name = str(quarter) ## powinno byc Q3
        print(var_name)
        assigned_blocks = unique_blocks[:3]
        print(assigned_blocks)
        for date_block in assigned_blocks:
            print(date_block)
            MQ_Dict[date_block] = var_name
        unique_blocks = unique_blocks[3:]
        print(unique_blocks)

### Adding quarter column

train_data['date_quarter'] = train_data['date_block_num'].map(MQ_Dict)
train_data = pd.concat([train_data, pd.get_dummies(train_data['date_quarter'])], axis=1)

train_data.columns = ['date', 'date_block_num', 'shop_id', 'item_id', 'item_price',
       'item_cnt_day', 'item_name', 'item_category_id','item_category_name', 'date_quarter', 'Q1',
       'Q2', 'Q3', 'Q4']
### Groupby data for the model input

agg_train = train_data.groupby(['date_block_num', 'shop_id', 'item_id', 'date_quarter',
                                'Q1', 'Q2', 'Q3', 'Q4']).agg({'item_cnt_day': ['sum']})

### Creating train and test data 

X_train = np.array(list(map(list, agg_train.index.values)))
y_train = agg_train.values

X_train = X_train.astype(np.float)


train_data[train_data['date_block_num'] == train_data['date_block_num'].max()] ## The last month is October 2015 with index 33 and Q4

test_data['date_block_num'] = train_data['date_block_num'].max() + 1
test_data['date_quarter'] = 4
X_test = test_data[['date_block_num', 'shop_id', 'item_id']].values

#X_train_dummies = pd.concat([pd.get_dummies(X_train[:,0]),pd.get_dummies(X_train[:,1]),pd.get_dummies(X_train[:,2], columns = col_first_first)], axis=1)
#items = np.unique(X_train[:,2])
#col_first, col_second, col_third = np.split(items,3)

### One Hot Endcoding 

oh0 = OneHotEncoder(categories='auto').fit(X_train[:,0].reshape(-1, 1)) ###!### categories = auto ? 
x0 = oh0.transform(X_train[:,0].reshape(-1, 1)) ###!### is this necessary?

oh1 = OneHotEncoder(categories='auto').fit(X_train[:,1].reshape(-1, 1))
x1 = oh1.transform(X_train[:,1].reshape(-1, 1))
x1_t = oh1.transform(X_test[:,1].reshape(-1, 1))

### Attempt with 4 different regression types ###

dmy = DummyRegressor().fit(X_train, y_train)
reg = LinearRegression().fit(X_train, y_train)

print ("Random Forest Fit execution start datetime: ")
print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

rfr = RandomForestRegressor(n_estimators = 80, n_jobs = -1).fit(X_train, y_train.ravel())

print ("Random Forest Fit execution end datetime: ")
print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

svr = SVR(kernel = 'rbf')  ###!### SVR not working
svr.fit(X_train,y_train.ravel()) 

### Setup of the Aritficial Neural Network

start_time = datetime.datetime.now()
print ("Artificial Neural Network Build start datetime : ")
print (start_time.strftime("%Y-%m-%d %H:%M:%S"))

regressor = Sequential()

#1 Hidden Layer    
regressor.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'softmax', input_dim = 95)) ### Relu = Rectifier ### Input_dim okreslamy tylko dla pierwszej warstwy, zeby algorytm sie pokapowal, potem juz sobie sam zalapie
#2 Hidden Layer 
regressor.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'softmax')) ### init = random, bo chcemy na początek przyznać wagom rozmiary bliskie 0 w opcji losowej
#3 HL
regressor.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'softmax')) ### init = random, bo chcemy na początek przyznać wagom rozmiary bliskie 0 w opcji losowej
#4 HL
regressor.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'softmax')) ### init = random, bo chcemy na początek przyznać wagom rozmiary bliskie 0 w opcji losowej
# Output Layer
regressor.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'hard_sigmoid')) ### przy zmiennej objasnianej majacej wiecej kategorii zmieniamy units i funkcje aktywujaca na softmax (sigmoid dla wielu zmiennych) 

###!### Chart of the ANN

from ann_visualizer.visualize import ann_viz;

ann_viz(regressor, title="Future Sales Artificial Neural Network")
            
# Compile ANN

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mean_squared_error']) ### Adam - jeden z rodzajow Stochastic Gradient Descent
                                                ### Logaritmic Loss - funkcja, ktora okresla koszt funkcji logistycznej. Tu binary_ lub categorical_crossentropy
                                                ### Metrics = potrzebujemy liste z jednym elementem
# Train ANN
                                                
regressor.fit(X_train_dummies, y_train, batch_size = 3, epochs = 1)

end_time = datetime.datetime.now()

print ("Artificial Neural Network Build end date time : ")
print (end_time.strftime("%Y-%m-%d %H:%M:%S"))

#### Root Mean Squared Errors for all regressors


rmse_dmy = np.sqrt(mean_squared_error(y_train, dmy.predict(X_train)))
print('Dummy RMSE: %.4f' % rmse_dmy)
rmse_reg = np.sqrt(mean_squared_error(y_train, reg.predict(X_train)))
print('Linear Regression RMSE: %.4f' % rmse_reg)
rmse_rfr = np.sqrt(mean_squared_error(y_train, rfr.predict(X_train)))
print('Random Forest Regression RMSE: %.4f' % rmse_rfr) ### 2.2034 before GS
rmse_ann = np.sqrt(mean_squared_error(y_train, regressor.predict(X_train_dummies)))
print('Artificial Neural Network RMSE: %.4f' % rmse_ann) ###!### Search for ANN for categorical variables - get_dummies(?)


#### NOTES ####

# Count total price of transactions 
train_data['total_price'] = train_data['item_price'] * train_data['item_cnt_day']

from sklearn.model_selection import GridSearchCV
reg = RandomForestRegressor()
grid_values = {'n_estimators': [100,200,300]}
grid_clf_acc = GridSearchCV(reg, param_grid = grid_values,scoring = 'neg_mean_squared_error')
grid_clf_acc.fit(X, y_train.ravel())

### Splitting the data into training and test set

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

### Extract data about a single shop with single_values function

def single_values(shop,item):

    single_td = single_td = train_data[train_data['shop_id'] == shop]
    single_td = single_td[single_td['item_id'] == item]
    single_td = single_td[['date', 'total_price']]
    
    return single_td
#    with plt.xkcd():
#        plt.subplot(211)
#        plt.bar(single_td['date'], single_td['total_price'])
#        plt.xlabel('Date')
#        plt.xticks(rotation='vertical')
#        plt.ylabel('Price of an item')
#        plt.gca().yaxis.set_minor_formatter(NullFormatter())
#        plt.title('Price of the item: ' + str(item) +' in shop: ' + str(shop))  
#        plt.subplot(212)
#        n, bins, patches = plt.hist(single_td['total_price'], 50, density=1, facecolor='b', alpha=0.75)
#        plt.xlabel('Prices')
#        plt.ylabel('Probability')
#        plt.gca().yaxis.set_minor_formatter(NullFormatter())
#        plt.title('Histogram of Prices')
#        plt.show()

# test = single_values(18, 20949)
    
df = pd.DataFrame(test['total_price'], index=test['date'], columns=["Values"])
autocorrelation_plot(df)

### Converting date to datetime and adding day of week

train_data['date'] = pd.to_datetime(train_data['date'])
train_data['date_of_week'] = train_data['date'].dt.dayofweek ## 0 = Monday, 6 = Sunday
train_data['date_year_month']= pd.to_datetime(train_data['date']).dt.to_period('M') ## adding YYYYMM for groupby

end_time = time.time()
print(end_time-start_time) ### 5 minut

### Reorder coluns from most to least important

column_order = ['shop_id', 'item_id', 'item_cnt_day', 'date', 'item_name', 
                'date_of_week', 'date_year_month', 'item_price',
                'item_category_name', 'item_category_id', 'date_block_num']
train_data = train_data[column_order]
sum_per_shop = train_data[['shop_id', 'item_cnt_day', 'date_year_month']].groupby(by=['shop_id', 'date_year_month']).agg({'item_cnt_day': ['sum']})
