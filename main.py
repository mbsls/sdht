import pandas as pd
import numpy as np
import seaborn as sns
import yaml
from fredapi import Fred
import functs as auxfun

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
            df_aux = fred.get_series_first_release(all_variables[variable]['code'])
            df_aux.index = pd.to_datetime(df_aux.index)
            df_aux.name = all_variables[variable]['label']
            df = pd.concat((df, df_aux),axis=1)
except:
    for variables in var_list:
        df_aux = fred.get_series(all_variables[variable]['code'])
        df_aux.index = pd.to_datetime(df_aux.index)
        df_aux.name = all_variables[variable]['label']
        df = pd.concat((df, df_aux),axis=1)

# df = auxfun.transform_data(df, all_variables)
df.to_csv('a.csv')