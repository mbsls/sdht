import pandas as pd
import numpy as np
import datetime

def transform_data(df, variable_dict,sample_start=None, sample_end=None):
    for c in df.columns:
        dictionary = variable_dict[c]
        try:
            if dictionary['accumulate'] == 'period-over-period':
                var = df.loc[:, c]
                aux = np.ones((len(var),))
                aux[0] = 100
                for j in range(1, len(var)):
                    if np.isfinite(var.iloc[j]):
                        aux[j] = aux[j-1] * (1+var.iloc[j]/100)
                    else:
                        aux[j] = aux[j-1]
                df.loc[:, c] = aux
        except:
            pass


        try:
            if dictionary['transformation'] == 'log':
                df.loc[:, c] = 100 * np.log(df.loc[:,c])
            elif dictionary['transformation'] == 'yoy-m':
                df.loc[:, c] = 100 * (df.loc[:, c]-df.loc[:, c].shift(12))/df.loc[:, c].shift(12)
            elif dictionary['transformation'] == 'yoy-q':
                df.loc[:, c] = 100 * (df.loc[:, c]-df.loc[:, c].shift(4))/df.loc[:, c].shift(4)
            elif dictionary['transformation'] == 'd12':
                df.loc[:, c] = df.loc[:, c]-df.loc[:, c].shift(12)
        except:
            pass
        

        df.loc[:,c] = pd.to_numeric(df.loc[:,c], errors='coerce')
        try:
            df.loc[:,c] = df.loc[:,c].interpolate(dictionary['interpolate'])
        except:
            pass
        
    # last adjustments here..
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.asfreq('MS')

    
    if isinstance(sample_start, datetime.datetime):
        df = df.loc[df.index >= sample_start]
    elif (sample_start is not None) & (not isinstance(sample_start, datetime.datetime)):
        print('Sample start date must be a datetime!')
        return -1
    
    
    if isinstance(sample_end, datetime.datetime):
        df = df.loc[df.index <= sample_end]
    elif (sample_end is not None) & (not isinstance(sample_end, datetime.datetime)):
        print('Sample end date must be a datetime!')
        return -1
    
    return df

def inverse_transform(df, variable_dict):

    for c in df.columns:
        dictionary = variable_dict[c]

        try:
            if dictionary['inverse_transform'] == True:
                if dictionary['transformation'] == 'log':
                    df.loc[:, c] = np.exp(df.loc[:,c]/100)

        except:
            pass

    return df