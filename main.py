import pandas as pd
import numpy as np
import seaborn as sns
import yaml
from fredapi import Fred
import functs as auxfun
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore future warnings...

# Load config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Download data
api_key = config['api_key']
fred = Fred(api_key=config['api_key'])

var_list = config['data_series'].keys()
all_variables = config['data_series']

df = pd.DataFrame()
try:
    if config['real_time']:
        
        for variable in var_list:
            try:
                if all_variables[variable]['revision'] == 'get_rate':
                        print('Now downloading variable: ' + variable + ', which is real time in rates.')
                
                # basic scheme... # this is PERIOD OVER PERIOD
                df_aux = fred.get_series_first_release(all_variables[variable]['code'])
                df_aux.index = pd.DatetimeIndex(pd.to_datetime(df_aux.index), freq='MS')
                df_aux.sort_index(ascending=True)
                df_aux = 100 * (df_aux-df_aux.shift(1))/df_aux.shift(1)

                dates_to_download = fred.get_series_vintage_dates(all_variables[variable]['code'])
                for date in dates_to_download:
                    datestr = date.strftime('%m-%d-%Y')
                    print('Now downloading: ' + datestr)
                    df_aux2 = fred.get_series_as_of_date(all_variables[variable]['code'], datestr)
                    df_aux[pd.to_datetime(df_aux2.iloc[-1].date)] = 100 * (df_aux2.iloc[-1].value - df_aux2.iloc[-2].value)/(df_aux2.iloc[-2].value)

                df_aux.index = pd.to_datetime(df_aux.index)
                df_aux.sort_index(ascending=True)
                try:
                    df_aux.index = pd.DatetimeIndex(df_aux.index, freq='MS')
                except:
                    df_aux.index = pd.DatetimeIndex(df_aux.index)# + pd.DateOffset(months=1)
                df_aux.sort_index(ascending=True)
                df_aux.name = variable
                df = pd.concat((df, df_aux),axis=1)

            except:
                print('Now downloading variable: ' + variable)
                df_aux = fred.get_series_first_release(all_variables[variable]['code'])
                df_aux.index = pd.to_datetime(df_aux.index)
                df_aux.sort_index(ascending=True)
                try:
                    df_aux.index = pd.DatetimeIndex(df_aux.index, freq='MS')
                except:
                    df_aux.index = pd.DatetimeIndex(df_aux.index)# + pd.DateOffset(months=1)
                df_aux.sort_index(ascending=True) # Just making sure
                df_aux.name = variable
                df = pd.concat((df, df_aux),axis=1)
except:
    for variables in var_list:
        df_aux = fred.get_series(all_variables[variable]['code'])
        df_aux.index = pd.to_datetime(df_aux.index)
        df_aux.sort_index(ascending=True)
        df_aux.index = pd.DatetimeIndex(df_aux.index, freq='MS')
        df_aux.name = variable
        df = pd.concat((df, df_aux),axis=1)

sample_start = pd.to_datetime(config['sample_start'])
df = auxfun.transform_data(df, all_variables, sample_start=sample_start)
model = VAR(df)