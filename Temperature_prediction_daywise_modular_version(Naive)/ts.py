
######################################### Time series forcasting(main_script) #############################################

import pandas as pd          
import numpy as np          # For mathematical calculations 
import matplotlib.pyplot as plt  # For plotting graphs 
from datetime import datetime    # To access datetime 
from pandas import Series        # To work on series 
#%matplotlib inline 
import warnings                   # To ignore the warnings warnings.filterwarnings("ignore")
import ts_ts_imutils
from matplotlib.pylab import rcParams
import cl
import train_test_valid_gen

if __name__ == '__main__':

    
    train_0,test_0,valid_0,train_original_0,test_original_0,valid_original_0 = ts_imutils.load_data()
    #print(train)
    '''
    ts_imutils.look_data(train_0,test_0)
    plt.show()
    '''
    train_1,valid_1,test_1 = ts_imutils.clean_NaN_dataset(train_0,valid_0,test_0)
    train_original_1,valid_original_1,test_original_1 = ts_imutils.clean_NaN_dataset(train_original_0,valid_original_0,test_original_0)
    #print("\n There is : ",test_1.isnull().sum())

    '''
    train_1,test_1,train_original_1,test_original_1 = ts_imutils.feature_extract(train_0,test_0,train_original_0,test_original_0)
    plt.show()

    ts_imutils.exploratory_analysis(train_1)
    plt.show()

    train_2,test_2 = ts_imutils.model(train_1,test_1)
    plt.show()

    train_3,valid_3 = ts_imutils.train_vadidation_set(train_2)
    plt.show()
    '''

    #For matching the input arguments for the next function
    train_3,valid_3 = train_1,valid_1

    ts_imutils.modelling_naive(train_3,valid_3)
    plt.show() #you use it inside the function also for all the cases as used in modelling_naive()

    ts_imutils.modelling_rolling_avg(train_3,valid_3)

    ts_imutils.modelling_simple_exponential_smoothing(train_3,valid_3)
    

    # Not worked
    #fit1,y_hat_avg = ts_imutils.modelling_Holt_Linear_Trend_Model(train_3,valid_3)

    # Will not work
    #temp2 = ts_imutils.predictions_holt_linear_trend_model(fit1,train_3,test_2,train_original_1,test_original_1)

    # Will not work
    #ts_imutils.Holt_Winter_model_on_daily_time_series(train_3,test_2,test_original_1,valid_3,temp2) 

    ts_imutils.test_stationarity(train_original_1['Temperature'])

    # Not working
    Train_log,train_log_diff = ts_imutils.removing_trend_log_transformation(train_3,valid_3)
    print("\n There is : ",Train_log.isnull().sum())

    # working
    #ts_imutils.removing_seasonality(Train_log)

    # working
    #ts_imutils.forecasting_ACF_PACF(train_log_diff)

    # Not working fully
    #ts_imutils.AR_model(Train_log,train_log_diff,valid_3)
    
    # Not working fully
    #ts_imutils.MA_model(Train_log,train_log_diff,valid_3)

    # Not working fully
    results_ARIMA = ts_imutils.combined_model(Train_log,train_log_diff,valid_3)

    #Not working
    #Let’s predict the values for validation set.
    #ARIMA_predict_diff=results_ARIMA.predict(start="2014-06-25", end="2014-09-25")
    #ARIMA_predict_diff=results_ARIMA.predict(start="2013-03-24", end="2013-09-24")
    #check_prediction_diff(ARIMA_predict_diff, valid_1)

    '''
    # Will not working
    ts_imutils.SARIMAX_model(train_3,valid_3,y_hat_avg,test_2,test_original_1,temp2)

    '''
