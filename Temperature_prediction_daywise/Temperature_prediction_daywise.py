import pandas as pd          
import numpy as np
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.pyplot as plt 
from datetime import datetime    
from pandas import Series        
import warnings            
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
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
from statsmodels.tsa.arima_model import ARMA
import itertools



def data_load_first(the_file):
    train = pd.read_csv(the_file,sep=r'\s*,\s*',header=0, encoding='ascii', engine='python')
    train_original=train.copy()
    return train,train_original

def seeing_column_name_first(train):
    for col in train.columns: 
        print(col) 
    print("\n")

def seeing_empty_value_first(train):
    null_columns=train.columns[train.isnull().any()]
    print("\n")
    print(train[train["Temperature"].isnull()][null_columns])
    print("\n")
    return train

def clean_nothing_dataset_first(train):
    print(train.isnull().sum())
    print("\n")
    train = train.dropna()
    print(train.isnull().sum())
    print("\n")
    return train

def removing_column_first(train):
    to_drop = ['_fog','_hail','_dewptm','_hum','_conds','_pressurem','_wspdm','_rain','_snow','_thunder','_tornado','_wgustm','_windchillm','_wspdm','_wdire','_wdird','_vism','_precipm','_heatindexm']
    train.drop(to_drop,inplace=True,axis=1)
    return train

def renaming_column_first(train):
    new_name = {'datetime_utc':'Datetime_UTC','_tempm':'Temperature',' ':'ID'}
    train.rename(columns = new_name,inplace = True)
    print(train.head(5))
    print("\n")
    return train

def replacing_column_values_first(train):
    replace_values = {'Smoke' :4,'Clear':0,'Haze':2,'Unknown':-2,'Scattered cloud':0,'Shallow Fog':-1,'Mostly Cloudy':3}
    train = train.replace({"Cloud_condition":replace_values})
    display(train.head(100))
    print("\n")
    return train

def save_modified(train,op_file):
    train.to_csv(op_file)
    return train

def load_data():

    train=pd.read_csv('temp.csv') 
    test=pd.read_csv('temp_test.csv')
    valid=pd.read_csv('temp_valid.csv')

    train_original=train.copy() 
    test_original=test.copy()
    valid_original=valid.copy()
    return train,test,valid,train_original,test_original,valid_original

def stationarity_check(ts):
    roll_mean = pd.Series(ts).rolling(window=12).mean()
    plt.plot(ts, color='green',label='Original')
    plt.plot(roll_mean, color='blue', label='Rolling Mean')
    plt.legend(loc='best')
    plt.title('Rolling Mean')
    plt.show()
    print('Augmented Dickey-Fuller test:')
    df_test = adfuller(ts)
    print("type of df_test: ",type(df_test))
    print("df_test: ",df_test)
    df_output = pd.Series(df_test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    print("df_output: \n",df_output)
    print("\n")
    for key,value in df_test[4].items():
        df_output['Critical Value (%s)'%key] = value
    print(df_output)

def train_vadidation_test_set_generation(train):

    Train=train.ix['1996-11-01':'2016-01-01']
    test=train.ix['2016-01-01':'2016-11-30']
    Train.Temperature.plot(figsize=(15,8), title= 'Daily Temperature', fontsize=14, label='train')
    test.Temperature.plot(figsize=(15,8), title= 'Daily Temperature', fontsize=14, label='test') 
    plt.xlabel("Datetime_UTC") 
    plt.ylabel("Passenger count") 
    plt.legend(loc='best') 
    plt.show()

    return Train,test



if __name__ == '__main__':

    # Data cleaning and prepration
    '''
    train_0,train_original_0 = data_load_first("Weather_data.csv")
    seeing_column_name_first(train_0)
    train_1 = removing_column_first(train_0)
    train_2 = renaming_column_first(train_1)
    train_3 = clean_nothing_dataset_first(train_2)
    train_4 = save_modified(train_3,'temp.csv')
    ###### Put the column name ID at the first column header of the temp.csv manually
    '''


    # For making a day wise data from the hour wise data from temp.csv and save it as temp1.csv
    '''
    df = pd.read_csv('temp.csv', index_col=[0], parse_dates=[0], usecols=[0,1])
    df = df.resample('d').mean()
    df = save_modified(df,'temp1.csv')
    '''


    # Prediction part(Exploratory analysis,modelling and forecasting)

    df=pd.read_csv('temp1.csv')
    print(type(df))
    print(df.describe())
    df.index = pd.to_datetime(df.Datetime_UTC)
    print(df)
    df = df.drop('Datetime_UTC', axis=1)
    print(df)
    df = df.loc['1996-11-01':]
    print(df)
    print(df[df.isnull()])
    print(len(df[df.isnull()]))
    df = df.sort_index()
    print(df.index)
    df.Temperature.fillna(method='pad', inplace=True)
    print(df[df.Temperature.isnull()])
    print(df.describe())
    df['Total_days'] = range(0,len(df.index.values))
    print(df.head(10))
    print(df.tail(10))
    fig = plt.figure(1)
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('Total_days')
    ax1.set_ylabel('Temperature')
    ax1.set_title('Original Plot')
    ax1.plot('Total_days', 'Temperature', data = df);
    plt.show()
    print(df)
    stationarity_check(df.Temperature)
    df['Roll_Mean'] = pd.Series(df.Temperature).rolling(window=12).mean()    
    df = save_modified(df,'temp_clean.csv')
    df_Tr,df_Te = train_vadidation_test_set_generation(df)
    print(df_Tr.head(40))
    plot_acf(df_Tr.Temperature, lags=50)
    plot_pacf(df_Tr.Temperature, lags=50)
    plt.xlabel('lags')
    plt.show()
    p = q = range(0, 4)
    pq = itertools.product(p, q)
    for param in pq:
        try:
            mod = ARMA(df_Tr.Temperature,order=param)
            results = mod.fit()
            print('ARMA{} - AIC:{}'.format(param, results.aic))
        except:
            continue
    model = ARMA(df_Tr.Temperature, order=(2,3))
    results_MA = model.fit()  
    plt.plot(df_Tr.Temperature)
    plt.plot(results_MA.fittedvalues, color='red')
    plt.title('Fitting data _ MSE: %.2f'% (((results_MA.fittedvalues-df_Tr.Temperature)**2).mean()))
    plt.show()
    predictions = results_MA.predict('2016/01/01', '11/30/2016')
    print(predictions)

