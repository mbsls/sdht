import pandas as pd
import numpy as np

def transform_data(df, variable_dict):
    for c in df.columns:
        dictionary = variable_dict[c]

        try:
            if dictionary['accumulate'] == 'period_over_period':
                var = df.loc[:, c]
                aux = np.ones((len(var),))
                aux[0] = 100
                for j in range(1, len(var)):
                    aux[j] = aux[j-1] * (1+var.iloc[j])
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
        except:
            pass

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