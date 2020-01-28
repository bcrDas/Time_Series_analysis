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
import imutils
import cl


def load_data():

    train=pd.read_csv("temp.csv") 

    train_original=train.copy()

    print(train.isnull().sum())

    #Used for generating train,validation and test set from temp.csv(First step if data is not fitting with our alogrithm model)
    return train,train_original  


# Used for generating train,validation and test set from temp.csv(First step if data is not fitting with our alogrithm model)
def feature_extract(train,train_original):

    train['Datetime_UTC'] = pd.to_datetime(train.Datetime_UTC,format='%Y%m%d-%H:%M') 
    train_original['Datetime_UTC'] = pd.to_datetime(train_original.Datetime_UTC,format='%Y%m%d-%H:%M')
    
    for i in (train,train_original):
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

    return train,train_original


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


# Used for generating train,validation and test set from temp.csv(First step if data is not fitting with our alogrithm model)
def model(train):

    train=train.drop('ID',1)
    train.Timestamp = pd.to_datetime(train.Datetime_UTC,format='%Y%m%d-%H:%M') 
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

    #test.Timestamp = pd.to_datetime(test.Datetime,format='%d-%m-%Y %H:%M') 
    #test.index = test.Timestamp  

    # Converting to daily mean 
    #test = test.resample('D').mean() 

    train.Timestamp = pd.to_datetime(train.Datetime_UTC,format='%Y%m%d-%H:%M') 
    train.index = train.Timestamp 
    # Converting to daily mean 
    train = train.resample('D').mean()

    return train


# Used for generating train,validation and test set from temp.csv(First step if data is not fitting with our alogrithm model)
def train_vadidation_test_set_generation(train):

    Train=train.ix['1996-11-01':'2013-09-24'] 
    valid=train.ix['2013-09-25':'2014-09-25']
    test=train.ix['2014-09-26':'2016-11-30']
    Train.Temperature.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14, label='train') 
    valid.Temperature.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14, label='valid') 
    test.Temperature.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14, label='valid')
    plt.xlabel("Datetime_UTC") 
    plt.ylabel("Passenger count") 
    plt.legend(loc='best') 
    plt.show()

    return Train,valid,test


def save_modified(train,op_file):
    # Saving modified dataframe.
    train.to_csv(op_file)
    return train





if __name__ == '__main__':

    # Used for generating train,validation and test set from temp.csv(First step if data is not fitting with our alogrithm model)
    train_0,train_original_0 = load_data()
   
    train_1,train_original_1 = feature_extract(train_0,train_original_0)
    plt.show()

    train_2 = model(train_1)
    plt.show()

    train_3,valid_3,test_3 = train_vadidation_test_set_generation(train_2)
    plt.show()

    train_3 = save_modified(train_3,'temp_train.csv')
    valid_3 = save_modified(valid_3,'temp_valid.csv') 
    #### Please rename the file to aviode problem(because I faced,don't know the reason)
    test_3 = save_modified(test_3,'temp_test.csv')

    # Just for checking NaN is present or not
    print(train_3.isnull().sum())
    #train_3 = train_3.dropna()
    train_3= train_3.fillna(0)
    print(train_3.isnull().sum())
