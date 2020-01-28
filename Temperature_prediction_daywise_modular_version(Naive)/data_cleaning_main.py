

############################################### Data cleaning (main_script) #########################################

import pandas as pd
import numpy as np 
from IPython.display import display
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
import data_cleaning


if __name__ == '__main__':


    # Data(.csv) : Load,modify,save
    train_0,train_original_0 = data_cleaning.data_load("Weather_data.csv")

    data_cleaning.seeing_column_name(train_0)

    train_1 = data_cleaning.removing_column(train_0)

    train_2 = data_cleaning.renaming_column(train_1)

    train_3 = data_cleaning.clean_nothing_dataset(train_2)

    train_4 = data_cleaning.save_modified(train_3,'temp.csv')

    ###### Put the column name ID at the first column header of the temp.csv

   

    






































    '''
    train['datetime_utc'] = pd.to_datetime(train.datetime_utc,format='%Y%m%d-%H:%M') 
    train_original['datetime_utc'] = pd.to_datetime(train_original.datetime_utc,format='%Y%m%d-%H:%M')

    for i in (train,train_original):

        i['year']=i.datetime_utc.dt.year 
        i['month']=i.datetime_utc.dt.month 
        i['day']=i.datetime_utc.dt.day
        i['Hour']=i.datetime_utc.dt.hour 

    train['day of week']=train['datetime_utc'].dt.dayofweek 
    temp = train['datetime_utc']
    train['weekend']=temp

    train.index = train['datetime_utc'] # indexing the Datetime to get the time period on the x-axis. 
    df=train.drop('_rain',1)           # drop ID variable to get only the Datetime on x-axis. 
    ts = df['_tempm'] 
    plt.figure(figsize=(16,8)) 
    plt.plot(ts, label='Temperature Variation') 
    plt.title('Time Series') 
    plt.xlabel("Time(year-month)") 
    plt.ylabel("Temperature") 
    plt.legend(loc='best')
    plt.show()
    '''



    '''
    #Exploratory_analysis
    
    train.groupby('year')['_tempm'].mean().plot.bar()
    plt.show()
    

    train.groupby('month')['_tempm'].mean().plot.bar()
    plt.show()

    temp=train.groupby(['year', 'month'])['_tempm'].mean() 
    temp.plot(figsize=(15,5), title= 'Temperature(Monthwise)', fontsize=14)
    plt.show()

    train.groupby('day')['_tempm'].mean().plot.bar()
    plt.show()

    train.groupby('Hour')['_tempm'].mean().plot.bar()
    plt.show()

    #train.groupby('weekend')['_tempm'].mean().plot.bar()
    #plt.show()

    train.groupby('day of week')['_tempm'].mean().plot.bar()
    plt.show()

    train.groupby('_dewptm')['_tempm'].mean().plot.bar()
    plt.show()

    train.groupby('_hum')['_tempm'].mean().plot.bar()
    plt.show()

    train.groupby('_pressurem')['_tempm'].mean().plot.bar()
    plt.show()
    '''

