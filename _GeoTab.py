import pandas as pd, numpy as np, matplotlib.pyplot as plt
import json, os
from sklearn.preprocessing import OneHotEncoder
os.chdir("C:\\Users\\Marek.Pytka\\Desktop\\Inne szkolenia\\GeoTab\\")


train_data = pd.read_csv('C:\\Users\\Marek.Pytka\\Desktop\\Inne szkolenia\\GeoTab\\train.csv')
test_data = pd.read_csv('C:\\Users\\Marek.Pytka\\Desktop\\Inne szkolenia\\GeoTab\\test.csv')
sample_sub = pd.read_csv('C:\\Users\\Marek.Pytka\\Desktop\\Inne szkolenia\\GeoTab\\sample_submission.csv')

with open('submission_metric_map.json') as json_file:
    data = json.load(json_file)

for column in train_data[col_x_variables].columns:
    print(train_data[column].unique())
    
sample_sub.head()

col_x_variables = ['EntryHeading', 'ExitHeading', 'Hour', 'Weekend', 
                   'Month', 'City']
col_y_variables = ['TotalTimeStopped_p20', 'TotalTimeStopped_p50', 
                   'TotalTimeStopped_p80','DistanceToFirstStop_p20',
                   'DistanceToFirstStop_p50','DistanceToFirstStop_p80']

tts_percentiles = ['TotalTimeStopped_p20', 'TotalTimeStopped_p40',
                   'TotalTimeStopped_p50', 'TotalTimeStopped_p60', 
                   'TotalTimeStopped_p80']

tffs_percentiles = ['TimeFromFirstStop_p20', 'TimeFromFirstStop_p40',
                    'TimeFromFirstStop_p50', 'TimeFromFirstStop_p60',
                    'TimeFromFirstStop_p80']

dtfs_percentiles = ['DistanceToFirstStop_p20','DistanceToFirstStop_p40', 
                    'DistanceToFirstStop_p50','DistanceToFirstStop_p60', 
                    'DistanceToFirstStop_p80']

y_medians =  ['TotalTimeStopped_p50', 'TimeFromFirstStop_p50','DistanceToFirstStop_p50']

def distributions(row_id):

    print('Distribution of Time From First Stop:')
    print(train_data[tffs_percentiles].iloc[row_id,:])
    print('Distribution of Total Time Stopped:')
    print(train_data[tts_percentiles].iloc[row_id,:])
    print('Distribution of Distance to First Stop:')
    print(train_data[dtfs_percentiles].iloc[row_id,:])

distributions(17000)



from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
train_data['City'] = labelencoder.fit_transform(train_data['City'])

train_data[col_x_variables]