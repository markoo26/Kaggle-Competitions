### Import libraries ###
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller, pacf, acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools, webbrowser, ctypes, datetime
from pmdarima.arima.stationarity import ADFTest
from pmdarima.arima.utils import ndiffs
import time, sys

toolbar_width = 40


plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})
##C:\Users\Marek.Pytka\anaconda3\Lib\site-packages\statsmodels\graphics

### Definitions

ctypes.windll.user32.MessageBoxW(0, "Stationarity - Stationarity in a time series indicates that a series \n" +
"statistical attributes, such as mean, variance, etc., are constant over time \n" +
"(i.e., it exhibits low heteroskedasticity.", "Definition of Stationarity ",1)

# Important distinction between number of differences and lags

### Useful links

webbrowser.open_new_tab('https://www.kaggle.com/c/demand-forecasting-kernels-only/data')
webbrowser.open_new_tab('https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/')
webbrowser.open_new_tab('https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/')
webbrowser.open_new_tab('https://www.alkaline-ml.com/pmdarima/tips_and_tricks.html')

### Load data

#P

train_data = pd.read_csv('C:\\Users\\Marek.Pytka\\Desktop\\Inne szkolenia\\DemandForecasting\\train.csv')
test_data = pd.read_csv('C:\\Users\\Marek.Pytka\\Desktop\\Inne szkolenia\\DemandForecasting\\test.csv')

#C 

train_data = pd.read_csv('C:\\Users\\Marek\\Desktop\\Python\\Kaggle\\Demand Forecasting\\train.csv')
test_data = pd.read_csv('C:\\Users\\Marek\\Desktop\\Python\\Kaggle\\Demand Forecasting\\test.csv')

### Description of predictors

train_data.columns
train_data.store.unique() ### 10 stores
train_data.item.unique() ### 50 stores
train_data.date.unique() ### dates from '2013-01-01' to '2017-12-31'

### Check values for (single) shop & item

shops_total = train_data.store.unique().tolist()
items_total = train_data.item.unique().tolist()

### Dynamic assignment of ARIMA parameters

start_time = datetime.datetime.now()
print ("ARIMA parameters estimation for 10 shops and 45 items estimation start datetime : ")
print (start_time.strftime("%Y-%m-%d %H:%M:%S"))

arima_summary = pd.DataFrame(columns = ['Shop', 'Item', 'AR_Order', 'Diff_Order', 'MA_Order']) ### shop_id, item_id, p, d, q (AR-part, Differentiation, MA-part)
for i, j in itertools.product(range(1,len(shops_total)), range(1,len(items_total))):
   
    shops = [i]
    items = [j]
   
    data = train_data.loc[(train_data['store'].isin(shops)) & train_data['item'].isin(items)]
    data = data.reset_index()
    data.drop(['index'], axis=1)
    result_d0 = adfuller(data['sales'].dropna())[1]
    result_d1 = adfuller(data['sales'].diff().dropna())[1]
    result_d2 = adfuller(data['sales'].diff().diff().dropna())[1]

### assign d_parameter using the AD_Fuller test
   
    if result_d0 < 0.05:
        d_order = 0
    elif result_d1 < 0.05:
        d_order = 1
    elif result_d2 < 0.05:
        d_order = 2
       
### assign p_parameter using the PACF confidence interval
   
    if d_order == 0:
        value, con_int = pacf(data['sales'], alpha = 0.05)
    if d_order == 1:
        value, con_int = pacf(data['sales'].diff(), alpha = 0.05)
    if d_order == 2:
        value, con_int = pacf(data['sales'].diff().diff(), alpha = 0.05)
    left_bd = list(np.around((con_int[:, 0] - value),4))
    right_bd = list(np.around((con_int[:, 1] - value),4))
    intervals = pd.DataFrame(zip(left_bd, right_bd))
    series_pacf = pd.DataFrame(pacf(data['sales']))
   
    for value, index in zip(series_pacf.values, range(len(series_pacf.values))):
        if value > intervals[0][index] and value < intervals[1][index]:
            p_order = int(series_pacf[series_pacf[0]==float(value)].index.values)
            break
        else:
            p_order = 1
       
### assign q_parameter using the ACF confidence interval
   
    if d_order == 0:
        value, con_int = acf(data['sales'], alpha = 0.05)
    if d_order == 1:
        value, con_int = acf(data['sales'].diff(), alpha = 0.05)
    if d_order == 2:
        value, con_int = acf(data['sales'].diff().diff(), alpha = 0.05)
    left_bd = list(np.around((con_int[:, 0] - value),4))
    right_bd = list(np.around((con_int[:, 1] - value),4))
    intervals = pd.DataFrame(zip(left_bd, right_bd))
    series_acf = pd.DataFrame(pacf(data['sales']))
   
    for value, index in zip(series_acf.values, range(len(series_acf.values))):
        if value > intervals[0][index] and value < intervals[1][index]:
            q_order = int(series_acf[series_acf[0]==float(value)].index.values)
            break
        else:
            q_order = 1

    arima_summary.loc[len(arima_summary)] = [i, j, p_order, d_order, q_order]
    
end_time = datetime.datetime.now()
print ("Code execution end datetime : ")
print (end_time.strftime("%Y-%m-%d %H:%M:%S"))

final_forecasts = pd.DataFrame(columns = ['Shop', 'Item','No_Of_Prediction', 
                                          'Predicted_Value'])
forecasts_columns = ['Shop', 'Item']

arima_start_time = datetime.datetime.now()
print ("ARIMA models execution start datetime : ")
print (arima_start_time.strftime("%Y-%m-%d %H:%M:%S"))

    
for i in range(0,len(arima_summary)):

    
    arima_par_names = ['AR_Order', 'Diff_Order', 'MA_Order']
    arima_par_values = arima_summary[arima_par_names].iloc[[i]]
    arima_pv_tuple = [tuple(x) for x in arima_par_values.to_records(index=False)]

    model = ARIMA(data['sales'], order=arima_pv_tuple[0])
    model_fit = model.fit(disp=0)

    shops = int(arima_summary[forecasts_columns].iloc[i].values[0])
    items = int(arima_summary[forecasts_columns].iloc[i].values[1])

    forecasts = model_fit.predict()[0:90]
    print(forecasts)
    shop_item_forecasts = pd.DataFrame()
    shop_item_forecasts['Shop'] = list(np.repeat(shops,90))
    shop_item_forecasts['Item']= list(np.repeat(items,90))
    shop_item_forecasts['No_Of_Prediction'] = list(np.arange(0,90,1))
    shop_item_forecasts['Predicted_Value'] = list(forecasts)
    
    final_forecasts = final_forecasts.append(shop_item_forecasts)

    arima_current_time = datetime.datetime.now()
    arima_time_difference = (arima_current_time - arima_start_time).total_seconds()
    elapsed_time = len(arima_summary)/(i+1)*arima_time_difference
    arima_predicted_end_time = arima_current_time + datetime.timedelta(seconds=elapsed_time)

    print('Finished forecasting for ' + str(i) + ' out of ' + str(len(arima_summary)) + ' models')
    print ("ARIMA models predicted end datetime : ")
    print (arima_predicted_end_time.strftime("%Y-%m-%d %H:%M:%S"))
