
############################################### Data cleaning functions(script) #########################################

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



# Data(.csv) : Load,modify,save

def data_load(the_file):
    train = pd.read_csv(the_file,sep=r'\s*,\s*',header=0, encoding='ascii', engine='python')
    #print(train.columns.tolist())
    #print(train)
    train_original=train.copy()
    #display(train.head(10))
    return train,train_original


def seeing_column_name(train):
    # iterating the columns 
    for col in train.columns: 
        print(col) 
    print("\n")


def seeing_empty_value(train):
    #display(train[train.Temperature.notnull()])

    # Count the Null Columns
    null_columns=train.columns[train.isnull().any()]
    #print("Initial Null columns: ",train[null_columns].isnull().sum())
    print("\n")

    #Single Column Is Null
    print(train[train["Temperature"].isnull()][null_columns])
    print("\n")

    # replacing na values in college with No college 
    #print(train)
    print("\n")
    return train

    
def clean_nothing_dataset(train):
    '''
    assert isinstance(train, pd.DataFrame), "train needs to be a pd.DataFrame"
    train.dropna(inplace=True)
    indices_to_keep = ~train.isin([np.nan]).any(1)
    return train[indices_to_keep].astype(np.float64)
    '''
    ##train = train.replace('', np.nan, inplace=True)
    print(train.isnull().sum())
    print("\n")
    train = train.dropna()
    #train = train.fillna(0)
    print(train.isnull().sum())
    print("\n")
    return train


def removing_column(train):
    #Removing the unused or irrelevant columns
    to_drop = ['_fog','_hail','_dewptm','_hum','_conds','_pressurem','_wspdm','_rain','_snow','_thunder','_tornado','_wgustm','_windchillm','_wspdm','_wdire','_wdird','_vism','_precipm','_heatindexm']
    train.drop(to_drop,inplace=True,axis=1)
    return train


def renaming_column(train):
    #Renaming the column names as per our convenience.
    new_name = {'datetime_utc':'Datetime_UTC','_tempm':'Temperature',' ':'ID'}
    train.rename(columns = new_name,inplace = True)
    display(train.head(5))
    print("\n")
    #print(train)
    return train


def replacing_column_values(train):
    #Replacing the value of the rows if necessary.
    replace_values = {'Smoke' :4,'Clear':0,'Haze':2,'Unknown':-2,'Scattered cloud':0,'Shallow Fog':-1,'Mostly Cloudy':3}
    train = train.replace({"Cloud_condition":replace_values})
    display(train.head(100))
    print("\n")
    return train


def save_modified(train,op_file):
    # Saving modified dataframe.
    train.to_csv(op_file)
    return train

