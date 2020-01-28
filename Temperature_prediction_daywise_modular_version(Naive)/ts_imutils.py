
###################################### Time series forcasting functions(scripts) ########################################

import pandas as pd          
import numpy as np          # For mathematical calculations 

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import matplotlib.pyplot as plt  # For plotting graphs 
from datetime import datetime    # To access datetime 
from pandas import Series        # To work on series 
#%matplotlib inline 
import warnings                   # To ignore the warnings warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error 
from math import sqrt 
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt 
import statsmodels.api as sm 
from statsmodels.tsa.stattools import adfuller
from matplotlib.pylab import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.tsa.stattools import acf, pacf 
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm


########### Here Replace(for temp_...csv) -> [ Datetime -> Datetime_UTC and Count -> Temperature ]

def load_data():

    train=pd.read_csv('temp_train.csv') 
    test=pd.read_csv('temp_test.csv')
    valid=pd.read_csv('temp_valid.csv')

    train_original=train.copy() 
    test_original=test.copy()
    valid_original=valid.copy()

    return train,test,valid,train_original,test_original,valid_original
    #return train,train_original  # Used for generating train,validation and test set from temp.csv(First step if data is not fitting with our alogrithm model)


def look_data(train,test):

    print(train.columns)
    print("\n")
    print(test.columns)
    print("\n")
    print(train.dtypes)
    print("\n")
    print(test.dtypes)
    print("\n")
    print(train.shape)
    print("\n")
    print(test.shape)


def clean_NaN_dataset(train,valid,test):
    '''
    assert isinstance(train, pd.DataFrame), "train needs to be a pd.DataFrame"
    train.dropna(inplace=True)
    indices_to_keep = ~train.isin([np.nan]).any(1)
    return train[indices_to_keep].astype(np.float64)
    '''
    ##train = train.replace('', np.nan, inplace=True)

    # Clean NaN entries
    '''
    print(train.isnull().sum())
    print("\n")
    '''
    train = train.dropna()
    #train = train.fillna(0)
    '''    
    print(train.isnull().sum())
    print("\n")
    '''
    '''
    print(valid.isnull().sum())
    print("\n")
    '''
    valid = valid.dropna()
    #valid = valid.fillna(0)
    '''
    print(valid.isnull().sum())
    print("\n")
    '''
    '''
    print(test.isnull().sum())
    print("\n")
    '''
    test = test.dropna()
    #test = test.fillna(0)
    '''
    print(test.isnull().sum())
    print("\n")
    '''
    return train,valid,test


def feature_extract(train,test,train_original,test_original):

    train['Datetime_UTC'] = pd.to_datetime(train.Datetime_UTC,format='%d-%m-%Y %H:%M') 
    test['Datetime_UTC'] = pd.to_datetime(test.Datetime_UTC,format='%d-%m-%Y %H:%M') 
    test_original['Datetime_UTC'] = pd.to_datetime(test_original.Datetime_UTC,format='%d-%m-%Y %H:%M') 
    train_original['Datetime_UTC'] = pd.to_datetime(train_original.Datetime_UTC,format='%d-%m-%Y %H:%M')
    
    for i in (train, test, test_original, train_original):
        i['year']=i.Datetime_UTC.dt.year 
        i['month']=i.Datetime_UTC.dt.month 
        i['day']=i.Datetime_UTC.dt.day
        i['Hour']=i.Datetime_UTC.dt.hour 

    train['day of week']=train['Datetime_UTC'].dt.dayofweek 
    temp = train['Datetime_UTC']
    def applyer(row):
        if row.dayofweek == 5 or row.dayofweek == 6:
            return 1
        else:
            return 0 
    temp2 = train['Datetime_UTC'].apply(applyer) 
    train['weekend']=temp2

    
    train.index = train['Datetime_UTC'] # indexing the Datetime to get the time period on the x-axis. 
    df=train.drop('ID',1)           # drop ID variable to get only the Datetime on x-axis. 
    ts = df['Temperature'] 
    plt.figure(figsize=(16,8)) 
    plt.plot(ts, label='Passenger Count') 
    plt.title('Time Series') 
    plt.xlabel("Time(year-month)") 
    plt.ylabel("Passenger count") 
    plt.legend(loc='best')

    return train,test,train_original,test_original


def exploratory_analysis(train):

    #Exploratory_analysis
    train.groupby('year')['Temperature'].mean().plot.bar()
    #train.groupby('month')['Count'].mean().plot.bar()
    '''
    temp=train.groupby(['year', 'month'])['Count'].mean() 
    temp.plot(figsize=(15,5), title= 'Passenger Count(Monthwise)', fontsize=14)
    '''
    #train.groupby('day')['Count'].mean().plot.bar()
    #train.groupby('Hour')['Count'].mean().plot.bar()
    #train.groupby('weekend')['Count'].mean().plot.bar()
    #train.groupby('day of week')['Count'].mean().plot.bar()



def model(train,test):

    train=train.drop('ID',1)
    train.Timestamp = pd.to_datetime(train.Datetime_UTC,format='%d-%m-%Y %H:%M') 
    train.index = train.Timestamp 
    # Hourly time series 
    hourly = train.resample('H').mean() 
    # Converting to daily mean 
    daily = train.resample('D').mean() 
    # Converting to weekly mean 
    weekly = train.resample('W').mean() 
    # Converting to monthly mean 
    monthly = train.resample('M').mean()

    fig, axs = plt.subplots(4,1) 
    hourly.Temperature.plot(figsize=(15,8), title= 'Hourly', fontsize=14, ax=axs[0]) 
    daily.Temperature.plot(figsize=(15,8), title= 'Daily', fontsize=14, ax=axs[1]) 
    weekly.Temperature.plot(figsize=(15,8), title= 'Weekly', fontsize=14, ax=axs[2]) 
    monthly.Temperature.plot(figsize=(15,8), title= 'Monthly', fontsize=14, ax=axs[3]) 

    test.Timestamp = pd.to_datetime(test.Datetime_UTC,format='%d-%m-%Y %H:%M') 
    test.index = test.Timestamp  

    # Converting to daily mean 
    test = test.resample('D').mean() 

    train.Timestamp = pd.to_datetime(train.Datetime_UTC,format='%d-%m-%Y %H:%M') 
    train.index = train.Timestamp 
    # Converting to daily mean 
    train = train.resample('D').mean()

    return train,test



# Used for generating train,validation and test set from temp.csv(First step if data is not fitting with our alogrithm model)
def train_vadidation_test_set_generation(train):

    Train=train.ix['1996-11-01':'2013-09-24'] 
    valid=train.ix['2013-09-25':'2014-09-25']
    test=train.ix['2014-09-26':'2016-11-30']
    Train.Temperature.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14, label='train') 
    valid.Temperature.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14, label='valid') 
    plt.xlabel("Datetime_UTC") 
    plt.ylabel("Passenger count") 
    plt.legend(loc='best') 
    plt.show()

    return Train,valid,test

def modelling_naive(Train,valid):

    dd= np.asarray(Train.Temperature) 
    y_hat = valid.copy() 
    y_hat['naive'] = dd[len(dd)-1] 
    plt.figure(figsize=(12,8)) 
    plt.plot(Train.index, Train['Temperature'], label='Train') 
    plt.plot(valid.index,valid['Temperature'], label='Valid') 
    plt.plot(y_hat.index,y_hat['naive'], label='Naive Forecast') 
    plt.legend(loc='best') 
    plt.title("Naive Forecast") 
    plt.show()

    # checking the accuracy of our model on validation data set.
    rms = sqrt(mean_squared_error(valid.Temperature, y_hat.naive)) 
    print("\n Mean squared error is : ",rms)


def modelling_rolling_avg(Train,valid):

    y_hat_avg = valid.copy() 
    y_hat_avg['moving_avg_forecast'] = Train['Temperature'].rolling(10).mean().iloc[-1] # average of last 10 observations. 
    plt.figure(figsize=(15,5)) 
    plt.plot(Train['Temperature'], label='Train') 
    plt.plot(valid['Temperature'], label='Valid') 
    plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 10 observations') 
    plt.legend(loc='best') 
    plt.show() 
    y_hat_avg = valid.copy() 
    y_hat_avg['moving_avg_forecast'] = Train['Temperature'].rolling(20).mean().iloc[-1] # average of last 20 observations. 
    plt.figure(figsize=(15,5)) 
    plt.plot(Train['Temperature'], label='Train') 
    plt.plot(valid['Temperature'], label='Valid') 
    plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 20 observations') 
    plt.legend(loc='best') 
    plt.show() 
    y_hat_avg = valid.copy() 
    y_hat_avg['moving_avg_forecast'] = Train['Temperature'].rolling(50).mean().iloc[-1] # average of last 50 observations. 
    plt.figure(figsize=(15,5)) 
    plt.plot(Train['Temperature'], label='Train') 
    plt.plot(valid['Temperature'], label='Valid') 
    plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 50 observations') 
    plt.legend(loc='best') 
    plt.show()

    rms = sqrt(mean_squared_error(valid.Temperature, y_hat_avg.moving_avg_forecast)) 
    print("\n Mean squared error is : ",rms)


def modelling_simple_exponential_smoothing(Train,valid):

    y_hat_avg = valid.copy() 
    fit2 = SimpleExpSmoothing(np.asarray(Train['Temperature'])).fit(smoothing_level=0.6,optimized=False)
    y_hat_avg['SES'] = fit2.forecast(len(valid)) 
    plt.figure(figsize=(16,8)) 
    plt.plot(Train['Temperature'], label='Train') 
    plt.plot(valid['Temperature'], label='Valid') 
    plt.plot(y_hat_avg['SES'], label='SES') 
    plt.legend(loc='best') 
    plt.show()

    rms = sqrt(mean_squared_error(valid.Temperature, y_hat_avg.SES)) 
    print("\n Mean squared error is : ",rms)


def modelling_Holt_Linear_Trend_Model(Train,valid):

    sm.tsa.seasonal_decompose(Train.Temperature).plot() 
    result = sm.tsa.stattools.adfuller(Train.Temperature) 
    plt.show()

    y_hat_avg = valid.copy() 
    fit1 = Holt(np.asarray(Train['Temperature'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1) 
    y_hat_avg['Holt_linear'] = fit1.forecast(len(valid)) 
    plt.figure(figsize=(16,8)) 
    plt.plot(Train['Temperature'], label='Train') 
    plt.plot(valid['Temperature'], label='Valid') 
    plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear') 
    plt.legend(loc='best') 
    plt.show()

    rms = sqrt(mean_squared_error(valid.Temperature, y_hat_avg.Holt_linear)) 
    print("\n Mean squared error is : ",rms)

    return fit1,y_hat_avg


def  predictions_holt_linear_trend_model(fit1,train,test,train_original,test_original):

    submission=pd.read_csv("sample_submission.csv")
    predict=fit1.forecast(len(test))
    test['prediction']=predict

    # Calculating the hourly ratio of count 
    train_original['ratio']=train_original['Temperature']/train_original['Temperature'].sum() 

    # Grouping the hourly ratio 
    temp=train_original.groupby(['Hour'])['ratio'].sum() 

    # Groupby to csv format 
    pd.DataFrame(temp, columns=['Hour','ratio']).to_csv('GROUPby.csv') 

    temp2=pd.read_csv("GROUPby.csv") 
    temp2=temp2.drop('Hour.1',1) 

    # Merge Test and test_original on day, month and year 
    merge=pd.merge(test, test_original, on=('day','month', 'year'), how='left') 
    merge['Hour']=merge['Hour_y'] 
    merge=merge.drop(['year', 'month', 'Datetime','Hour_x','Hour_y'], axis=1) 
    # Predicting by merging merge and temp2 
    prediction=pd.merge(merge, temp2, on='Hour', how='left') 

    # Converting the ratio to the original scale 
    prediction['Temperature']=prediction['prediction']*prediction['ratio']*24 
    prediction['ID']=prediction['ID_y']

    #Let’s drop all other features from the submission file and keep ID and Count only.
    submission=prediction.drop(['ID_x', 'day', 'ID_y','prediction','Hour', 'ratio'],axis=1) 
    # Converting the final submission to csv format 
    pd.DataFrame(submission, columns=['ID','Temperature']).to_csv('Holt linear.csv')

    return temp2


def Holt_Winter_model_on_daily_time_series(Train,test,test_original,valid,temp2):

    y_hat_avg = valid.copy() 
    fit1 = ExponentialSmoothing(np.asarray(Train['Count']) ,seasonal_periods=7 ,trend='add', seasonal='add',).fit() 
    y_hat_avg['Holt_Winter'] = fit1.forecast(len(valid)) 
    plt.figure(figsize=(16,8)) 
    plt.plot( Train['Temperature'], label='Train') 
    plt.plot(valid['Temperature'], label='Valid') 
    plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter') 
    plt.legend(loc='best') 
    plt.show()

    rms = sqrt(mean_squared_error(valid.Count, y_hat_avg.Holt_Winter)) 
    print("\n Mean squared error is : ",rms)

    predict=fit1.forecast(len(test))

    test['prediction']=predict
    # Merge Test and test_original on day, month and year 
    merge=pd.merge(test, test_original, on=('day','month', 'year'), how='left') 
    merge['Hour']=merge['Hour_y'] 
    merge=merge.drop(['year', 'month', 'Datetime','Hour_x','Hour_y'], axis=1) 

    # Predicting by merging merge and temp2 
    prediction=pd.merge(merge, temp2, on='Hour', how='left') 

    # Converting the ratio to the original scale 
    prediction['Temperature']=prediction['prediction']*prediction['ratio']*24
    
    #Let’s drop all features other than ID and Count
    prediction['ID']=prediction['ID_y'] 
    submission=prediction.drop(['day','Hour','ratio','prediction', 'ID_x', 'ID_y'],axis=1) 

    # Converting the final submission to csv format 
    pd.DataFrame(submission, columns=['ID','Temperature']).to_csv('Holt winters.csv')













 
def test_stationarity(timeseries):
        #Determing rolling statistics
    rolmean = pd.Series(timeseries).rolling(window=24).mean() # 24 hours on each day

    rolstd = pd.Series(timeseries).rolling(window=24).std()

        #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    #Perform Dickey-Fuller test:
    print("\n")
    print ('Results of Dickey-Fuller Test:')
    #Problem_below
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

    plt.rcParams['figure.figsize'] = 20,10
    plt.show()


def removing_trend_log_transformation(Train,valid):

    Train_log = np.log(Train['Temperature']) 
    valid_log = np.log(valid['Temperature'])
    moving_avg = pd.Series(Train_log).rolling(window=24).mean()
    plt.plot(Train_log) 
    plt.plot(moving_avg, color = 'red') 
    plt.show()

    train_log_moving_avg_diff = Train_log - moving_avg
    train_log_moving_avg_diff.dropna(inplace = True)
    test_stationarity(train_log_moving_avg_diff)

    train_log_diff = Train_log - Train_log.shift(1) 
    print("\n There is : ",train_log_diff.isnull().sum())
    #train_log_diff = train_log_diff.fillna(0)
    #print("\n There is : ",train_log_diff.isnull().sum())
    #Problem_below
    test_stationarity(train_log_diff.dropna())
    #test_stationarity(train_log_diff)

    return Train_log,train_log_diff


def removing_seasonality(Train_log):

    #Train_log = Train_log.fillna(0)
    #Problem_below
    decomposition = seasonal_decompose(pd.DataFrame(Train_log).Temperature.values, freq = 24) 

    trend = decomposition.trend 
    seasonal = decomposition.seasonal 
    residual = decomposition.resid 

    plt.subplot(411) 
    plt.plot(Train_log, label='Original') 
    plt.legend(loc='best') 
    plt.subplot(412) 
    plt.plot(trend, label='Trend') 
    plt.legend(loc='best') 
    plt.subplot(413) 
    plt.plot(seasonal,label='Seasonality') 
    plt.legend(loc='best') 
    plt.subplot(414) 
    plt.plot(residual, label='Residuals') 
    plt.legend(loc='best') 
    plt.tight_layout() 
    plt.show()

    train_log_decompose = pd.DataFrame(residual) 
    train_log_decompose['date'] = Train_log.index 
    train_log_decompose.set_index('date', inplace = True) 
    train_log_decompose.dropna(inplace=True) 
    #Problem_below
    test_stationarity(train_log_decompose[0])


def forecasting_ACF_PACF(train_log_diff):

    lag_acf = acf(train_log_diff.dropna(), nlags=25) 
    lag_pacf = pacf(train_log_diff.dropna(), nlags=25, method='ols')

    #ACF and PACF plot
    plt.plot(lag_acf) 
    plt.axhline(y=0,linestyle='--',color='gray') 
    plt.axhline(y=-1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray') 
    plt.axhline(y=1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray')
    plt.title('Autocorrelation Function') 
    plt.show() 
    plt.plot(lag_pacf) 
    plt.axhline(y=0,linestyle='--',color='gray') 
    plt.axhline(y=-1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray') 
    plt.axhline(y=1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function') 
    plt.show()


def AR_model(Train_log,train_log_diff,valid):

    #train_log_diff = train_log_diff.fillna(0)
    #Train_log = Train_log.fillna(0)

    model = ARIMA(Train_log, order=(2, 1, 0))  # here the q value is zero since it is just the AR model
    results_AR = model.fit(disp=-1)  
    plt.plot(train_log_diff.dropna(), label='original') 
    plt.plot(results_AR.fittedvalues, color='red', label='predictions') 
    plt.legend(loc='best') 
    plt.show()
'''
#Not working
#AR_predict=results_AR.predict(start="2013-09-25", end="2014-09-25")
AR_predict=results_AR.predict(start="2013-03-24", end="2013-09-24")
AR_predict=AR_predict.cumsum().shift().fillna(0) 
AR_predict1=pd.Series(np.ones(valid.shape[0]) * np.log(valid['Count'])[0], index = valid.index) 
AR_predict1=AR_predict1.add(AR_predict,fill_value=0) 
AR_predict = np.exp(AR_predict1)
plt.plot(valid['Temperature'], label = "Valid") 
plt.plot(AR_predict, color = 'red', label = "Predict") 
plt.legend(loc= 'best') 
plt.title('RMSE: %.4f'% (np.sqrt(np.dot(AR_predict, valid['Temperature']))/valid.shape[0])) 
plt.show()
'''


def MA_model(Train_log,train_log_diff,valid):

    model = ARIMA(Train_log, order=(0, 1, 2))  # here the p value is zero since it is just the MA model 
    results_MA = model.fit(disp=-1)  
    plt.plot(train_log_diff.dropna(), label='original') 
    plt.plot(results_MA.fittedvalues, color='red', label='prediction') 
    plt.legend(loc='best') 
    plt.show()
'''
# Not working
#MA_predict=results_MA.predict(start="2014-06-25", end="2014-09-25")
MA_predict=results_MA.predict(start="2013-03-24", end="2013-09-24") 
MA_predict=MA_predict.cumsum().shift().fillna(0) 
MA_predict1=pd.Series(np.ones(valid.shape[0]) * np.log(valid['Temperature'])[0], index = valid.index) 
MA_predict1=MA_predict1.add(MA_predict,fill_value=0) 
MA_predict = np.exp(MA_predict1)
plt.plot(valid['Temperature'], label = "Valid") 
plt.plot(MA_predict, color = 'red', label = "Predict") 
plt.legend(loc= 'best') 
plt.title('RMSE: %.4f'% (np.sqrt(np.dot(MA_predict, valid['Temperature']))/valid.shape[0]))
plt.show()
'''
    


def combined_model(Train_log,train_log_diff,valid):

    model = ARIMA(Train_log, order=(2, 1, 2))  
    results_ARIMA = model.fit(disp=-1)  
    plt.plot(train_log_diff.dropna(),  label='original') 
    plt.plot(results_ARIMA.fittedvalues, color='red', label='predicted') 
    plt.legend(loc='best') 
    plt.show()
    return results_ARIMA


#Function can be used to change the scale of the model to the original scale.
def check_prediction_diff(predict_diff, given_set):
    predict_diff= predict_diff.cumsum().shift().fillna(0)
    predict_base = pd.Series(np.ones(given_set.shape[0]) * np.log(given_set['Temperature'])[0], index = given_set.index)
    predict_log = predict_base.add(predict_diff,fill_value=0)
    predict = np.exp(predict_log)

    plt.plot(given_set['Temperature'], label = "Given set")
    plt.plot(predict, color = 'red', label = "Predict")
    plt.legend(loc= 'best')
    plt.title('RMSE: %.4f'% (np.sqrt(np.dot(predict, given_set['Temperature']))/given_set.shape[0]))
    plt.show()

def check_prediction_log(predict_log, given_set):
    predict = np.exp(predict_log)
 
    plt.plot(given_set['Temperature'], label = "Given set")
    plt.plot(predict, color = 'red', label = "Predict")
    plt.legend(loc= 'best')
    plt.title('RMSE: %.4f'% (np.sqrt(np.dot(predict, given_set['Temperature']))/given_set.shape[0]))
    plt.show()



'''
#Let’s predict the values for validation set.
#ARIMA_predict_diff=results_ARIMA.predict(start="2014-06-25", end="2014-09-25")
ARIMA_predict_diff=results_ARIMA.predict(start="2013-03-24", end="2013-09-24")
#check_prediction_diff(ARIMA_predict_diff, valid)
        '''


def SARIMAX_model(Train,valid,y_hat_avg,test,test_original,temp2):

    y_hat_avg = valid.copy() 
    fit1 = sm.tsa.statespace.SARIMAX(Train.Temperature, order=(2, 1, 4),seasonal_order=(0,1,1,7)).fit() 
    y_hat_avg['SARIMA'] = fit1.predict(start="2014-6-25", end="2014-9-25", dynamic=True) 
    plt.figure(figsize=(16,8)) 
    plt.plot( Train['Temperature'], label='Train') 
    plt.plot(valid['Temperature'], label='Valid') 
    plt.plot(y_hat_avg['SARIMA'], label='SARIMA') 
    plt.legend(loc='best') 
    plt.show()

    rms = sqrt(mean_squared_error(valid.Count, y_hat_avg.SARIMA)) 
    print("\n Mean squared error is : ",rms)

    predict=fit1.predict(start="2014-9-26", end="2015-4-26", dynamic=True)

    test['prediction']=predict
    # Merge Test and test_original on day, month and year 
    merge=pd.merge(test, test_original, on=('day','month', 'year'), how='left') 
    merge['Hour']=merge['Hour_y'] 
    merge=merge.drop(['year', 'month', 'Datetime','Hour_x','Hour_y'], axis=1) 

    # Predicting by merging merge and temp2 
    prediction=pd.merge(merge, temp2, on='Hour', how='left') 

    # Converting the ratio to the original scale 
    prediction['Temperature']=prediction['prediction']*prediction['ratio']*24

    #Let’s drop all variables other than ID and Count
    prediction['ID']=prediction['ID_y'] 
    submission=prediction.drop(['day','Hour','ratio','prediction', 'ID_x', 'ID_y'],axis=1) 

    # Converting the final submission to csv format 
    pd.DataFrame(submission, columns=['ID','Temperature']).to_csv('SARIMAX.csv')