#### Import of the libraries ####

import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings, webbrowser, gc, os, datetime, time
warnings.simplefilter("ignore")

#### Models ####
from sklearn.ensemble import RandomForestRegressor

#### Change working directory

os.chdir("C:\\Users\\Marek\\Desktop\\Python\\Kaggle\\Datasets\\ASHRAE")

#### Fitting the data ####

full_train_data = pd.read_csv("full_train_data.csv")

train_y = full_train_data['meter_reading']
del full_train_data['meter_reading']
del full_train_data['timestamp']
train_X = full_train_data

for column in range(0,train_X.shape[1]):
    print(train_X.iloc[10:30,column])

np.unique(full_train_data['IsWeekend'].values)


#### Preprocess the data

## Monday = 0 and Sunday = 6

DICT_IsWeekend = { 
        '0':0,
        '1':0,
        '2':0,
        '3':0,
        '4':0,
        '5':1,
        '6':1,
        }

full_train_data['IsWeekend'] = full_train_data['IsWeekend'].map(DICT_IsWeekend)
np.unique(full_train_data['IsWeekend'].values)

### Code ###













####

#LGBM - fit_lgbm
#CatBoost Regressor - from catboost import CatBoostRegressor
#NGBoost - from ngboost.ngboost import NGBoost
#light GBM - import lightgbm as lgb

my_model = RandomForestRegressor()
my_model.fit(train_X, train_y)


