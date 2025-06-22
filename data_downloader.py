import pandas as pd
import numpy as np
import seaborn as sns
import yaml
from fredapi import Fred
import auxfun
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import datetime

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore future warnings...

def main(config_file):
    # Load config file
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Download data
    api_key = config['api_key']
    fred = Fred(api_key=config['api_key'])

    var_list = config['data_series'].keys()
    all_variables = config['data_series']

    name_convention = config['name_convention']
    try:
        real_time_flag = config['real_time']
    except:
        real_time_flag = None

    try:
        store_vintage = config['store_vintage']
    except:
        store_vintage = False

    df = pd.DataFrame()
    if real_time_flag == True:
        for variable in var_list:
            print('Now downloading variable: ' + variable)

            try:
                revision_flag = all_variables[variable]['revision']
            except:
                revision_flag = None

            # Start with latest value
            df_aux = fred.get_series_first_release(all_variables[variable]['code'])

            # Adjust index
            df_aux.index = pd.to_datetime(df_aux.index)
            df_frequency = df_aux.index.inferred_freq
            df_aux.index = pd.DatetimeIndex(df_aux.index, freq=df_frequency)
            df_aux.sort_index(ascending=True)
            df_aux = 100 * (df_aux-df_aux.shift(1))/df_aux.shift(1)
            
            if revision_flag == 'get_rate':
                dates_to_download = fred.get_series_vintage_dates(all_variables[variable]['code'])
                for date in dates_to_download:
                    datestr = date.strftime('%m-%d-%Y')
                    print('Now downloading ' + datestr + ' vintage')
                    df_aux2 = fred.get_series_as_of_date(all_variables[variable]['code'], datestr)

                    # Input rate
                    df_aux.loc[pd.to_datetime(df_aux2.iloc[-1].date)] = 100 * (df_aux2.iloc[-1].value - df_aux2.iloc[-2].value)/(df_aux2.iloc[-2].value)

            df_aux.index = pd.to_datetime(df_aux.index)
            df_aux.sort_index(ascending=True)

            if name_convention == 'label':
                df_aux.name = all_variables[variable]['label']
            else:
                df_aux.name = variable

            df = pd.concat((df, df_aux),axis=1)

    else:
        for variable in var_list:
            df_aux = fred.get_series(all_variables[variable]['code'])
            df_aux.index = pd.to_datetime(df_aux.index)
            df_aux.sort_index(ascending=True)
            df_aux.index = pd.DatetimeIndex(df_aux.index, freq='MS')
            
            if name_convention == 'label':
                df_aux.name = all_variables[variable]['label']
            else:
                df_aux.name = variable

            df = pd.concat((df, df_aux),axis=1)

    try:
        sample_start = pd.to_datetime(config['sample_start'])
    except:
        sample_start = pd.to_datetime('01-01-1700')

    try:
        sample_end = config['sample_end']
        sample_end = pd.to_datetime(sample_end)
    except:
        sample_end = pd.to_datetime('01-01-2100')

    df = auxfun.transform_data(df, all_variables, sample_start=sample_start, sample_end=sample_end)

    if store_vintage:
        lbl = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        df.index.name = 'date'
        df.to_csv(lbl + '.csv')

    return df

if __name__ == "__main__":
    config_file = 'config.yaml'
    df = main(config_file)
    print('Data downloaded successfully')