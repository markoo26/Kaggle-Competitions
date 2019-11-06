### TO DO:
#1. DISTPLOT przed i po uzupelnianiu nanow
#2. Convert to weekday

#### Import of the libraries ####

import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings, webbrowser, gc, os, datetime
warnings.simplefilter("ignore")


#### Models ####
from sklearn.ensemble import RandomForestRegressor


#### Change working directory

os.chdir("C:\\Users\\Marek.Pytka\\Desktop\\Inne szkolenia\\Skrypty\\ASHRAE")

#### Link to the data competition

#webbrowser.open_new_tab("https://www.kaggle.com/c/ashrae-energy-prediction")
#webbrowser.open_new_tab("https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard")
#### Data import ####

train_data = pd.read_csv("C:\\Users\\Marek.Pytka\\Desktop\\Inne szkolenia\\Skrypty\\ASHRAE\\train.csv")
test_data = pd.read_csv("C:\\Users\\Marek.Pytka\\Desktop\\Inne szkolenia\\Skrypty\\ASHRAE\\test.csv")
weather_train = pd.read_csv("C:\\Users\\Marek.Pytka\\Desktop\\Inne szkolenia\\Skrypty\\ASHRAE\\weather_train.csv")
weather_test = pd.read_csv("C:\\Users\\Marek.Pytka\\Desktop\\Inne szkolenia\\Skrypty\\ASHRAE\\weather_test.csv")
metadata = pd.read_csv("C:\\Users\\Marek.Pytka\\Desktop\\Inne szkolenia\\Skrypty\\ASHRAE\\building_metadata.csv")

train_data.head()
metadata.head()

#### Unique values per column ####

for col in train_data.columns:
    print(str(col) + ': ' + str(len(train_data[col].unique())) + ' unique values')

#### Predicted variables: chilled water, electric, hot water, and steam meters

#### Metadata - floor count check ####

# only less than 25% is available -> either ignore or fill the nans 
# by average per usage and year

missing_obs = metadata['floor_count'].isnull().sum()/len(metadata['floor_count'] + 0.0)*100
print('Percentage of missing observations for floor_count variable: ' + str(missing_obs))

missing_obs = metadata['year_built'].isnull().sum()/len(metadata['year_built'] + 0.0)*100
print('Percentage of missing observations for year_built variable: ' + str(missing_obs))

summary_primary_use_mod =  metadata.groupby(['primary_use'])['year_built'].mean()

### Replace null values in year_built by using mean year for each primary_use

for row, index in metadata.iterrows():
    if np.isnan(metadata['year_built'][row]):
        temp_value = summary_primary_use_mod[summary_primary_use_mod 
                                == metadata['primary_use'][row]]
        if len(temp_value) == 0: ## if no observations available - take mean of 
                                 ## the whole dataset 
            temp_value = np.nanmean(metadata['year_built']).astype(int)
                                ## else - get the mean for respective primary_use
        else:
            temp_value = int(temp_value[0]) 
            
        metadata['year_built'][row] = temp_value
     
### Create year bins ###

metadata['year_built_bin'] = pd.cut(metadata['year_built'], bins=4).astype(str)

#### Label Encoding the YBB variable ####

le_year = LabelEncoder()
le_year.fit(metadata['year_built_bin'])
metadata['year_built_bin'] = le_year.fit_transform(metadata['year_built_bin'])

#### Metadata - label encoding Primary Use variable ####

len(metadata['primary_use'].unique()) ### 16 categories

le = LabelEncoder()
le.fit(metadata['primary_use'])
metadata['primary_use'] = le.fit_transform(metadata['primary_use'])

#### Extract means for every primary_use and year_built_bin

summary_year = metadata.groupby(['year_built_bin'])['floor_count'].mean()
summary_primary_use = metadata.groupby(['primary_use'])['floor_count'].mean()

#### Replace null values in floor_count by using mean from mean per year built 
#### and mean per primary use

for index, row in metadata.iterrows():
    if metadata['floor_count'][index] is not int:
        temp_year_mean = summary_year[summary_year == metadata['year_built'][index]].values
        temp_pr_use_mean = summary_primary_use[summary_primary_use.index == metadata['primary_use'][index]].values
        
        if len(temp_year_mean) == 0:
            temp_year_mean = np.nan
        else: 
            temp_year_mean = temp_year_mean.astype(float)
        
        if len(temp_pr_use_mean) == 0:
            temp_pr_use_mean = np.nan
        else: 
            temp_pr_use_mean = temp_pr_use_mean[0].astype(float)
        
        temp_value = [temp_year_mean, temp_pr_use_mean]
        temp_value = np.nan_to_num(np.nanmean(temp_value))

        metadata['floor_count'][index] = int(temp_value)
    else:
        pass

#### Check missing observations ####
        
missing_obs = metadata['floor_count'].isnull().sum()/len(metadata['floor_count'] + 0.0)*100
print('Percentage of missing observations for floor_count variable: ' + str(missing_obs))

missing_obs = metadata['year_built'].isnull().sum()/len(metadata['year_built'] + 0.0)*100
print('Percentage of missing observations for year_built variable: ' + str(missing_obs))

#### Rebuild train & test datasets ####

## Merge columns ##

merge_columns = ['site_id','building_id', 'primary_use', 'square_feet', 
                 'year_built', 'floor_count']

mc_weather = ['site_id','timestamp','air_temperature', 'cloud_coverage']

#### Merge with metadata
merged_train_data = train_data.merge(metadata[merge_columns], left_on = 'building_id', 
                              right_on = 'building_id')

merged_test_data = test_data.merge(metadata[merge_columns], left_on = 'building_id', 
                              right_on = 'building_id')

#### Release memory ####

del train_data, test_data
gc.collect()

#### Output 1st merge to CSV ####

merged_train_data.to_csv("merged_train_data.csv",index_label=False)

## Merge with weather data 

## Drop unnecessary columns to reduce memory usage 

weather_train.drop(weather_train.columns.difference(mc_weather), 1, inplace=True)

## Create batches to avoid Memory Error

full_train_data = pd.DataFrame()
n_splits = 100
batch_size = int(merged_train_data.shape[0]/n_splits)

for i in range(1,n_splits+1):
    l_bound = (i - 1) * batch_size
    r_bound = i * batch_size
    print(l_bound)
    temp_batch = merged_train_data.iloc[l_bound:r_bound,:]
    temp_batch = temp_batch.merge(weather_train[mc_weather], 
                                                left_on = ['timestamp', 'site_id'],
                                                right_on = ['timestamp','site_id'])
    
    full_train_data = full_train_data.append(temp_batch)

#### Final Release of memory before analysis ####

del batch_size, col
del i, index, l_bound, mc_weather, merge_columns, merged_test_data
del merged_train_data, metadata, missing_obs, n_splits, r_bound, row
del summary_primary_use, summary_primary_use_mod, summary_year, temp_batch
del temp_pr_use_mean, temp_value, temp_year_mean, weather_test, weather_train
gc.collect()

#### Timestamp -> IsWeekend conversion and encoding
full_train_data['timestamp'] = pd.to_datetime(full_train_data['timestamp'])
full_train_data['IsWeekend'] = str(full_train_data['timestamp'].dt.dayofweek)

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



#### Fitting the data ####
train_y = full_train_data['meter_reading']
del full_train_data['meter_reading']
del full_train_data['timestamp']
train_X = full_train_data

my_model = RandomForestRegressor()
my_model.fit(train_X, train_y)




