###!### Need to be finished

#### IMPORT OF THE LIBRARIES ####

import pandas as pd, numpy as np, matplotlib.pyplot as plt
import webbrowser, time
from pandas.plotting import autocorrelation_plot

#### LINK TO THE COMPETITION

webbrowser.open_new_tab('https://www.kaggle.com/marcoracer/predict-future-sales-with-randomforest/notebook')

#### DATA PREPARATION 

start_time = time.time()
train_data = pd.read_csv('C:\\Users\\Marek\\Desktop\\Python\\Kaggle\\FutureSales\\sales_train_v2.csv')
items_categories = pd.read_csv('C:\\Users\\Marek\\Desktop\\Python\\Kaggle\\FutureSales\\item_categories.csv')
items = pd.read_csv('C:\\Users\\Marek\\Desktop\\Python\\Kaggle\\FutureSales\\items.csv')

### Converting date to datetime and adding day of week

train_data['date'] = pd.to_datetime(train_data['date'])
end_time = time.time()
print(end_time-start_time) ### 5 minut

### Join items and item categoreis

items = items.merge(items_categories, left_on='item_category_id', right_on='item_category_id', how='left')
train_data = train_data.merge(items, left_on ='item_id', right_on = 'item_id', how = 'left' )
train_data = train_data.sort_values(by = ['shop_id', 'item_id', 'date'])

### Reorder coluns from most to least important

column_order = ['shop_id', 'item_id', 'item_cnt_day', 'date', 'item_name', 'item_price',
                'item_category_name', 'item_category_id', 'date_block_num']
train_data = train_data[column_order]
sum_per_shop = train_data[['shop_id', 'item_cnt_day', 'date_year_month']].groupby(by=['shop_id', 'date_year_month']).agg({'item_cnt_day': ['sum']})

### Summarize count of rows per shop_id/item_id pair

summary = train_data.groupby(['shop_id', 'item_id']).agg({'item_id' :['count']})    
summary = summary.sort_values(by='date', ascending = False) ###!### Add title to the Series
        
test = single_values(18, 20949)

df = pd.DataFrame(test['total_price'], index=test['date'], columns=["Values"])
autocorrelation_plot(df)

### Splitting the data into training and test set

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



#### NOTES ####

train_data['date_of_week'] = train_data['date'].dt.dayofweek ## 0 = Monday, 6 = Sunday
train_data['date_year_month']= pd.to_datetime(train_data['date']).dt.to_period('M') ## adding YYYYMM for groupby

# Count total price of transactions 
train_data['total_price'] = train_data['item_price'] * train_data['item_cnt_day']

# Plot for every single shop:


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
