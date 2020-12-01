import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

def load_winequality_data():

    raw_data_file = os.path.join(os.path.dirname(__file__), 'winequality_raw.csv')
    processed_file = os.path.join(os.path.dirname(__file__), 'winequality_processed.csv')

    ##### White Wine Quality Data Processing
    raw_df = pd.read_csv(raw_data_file, sep=';')
    processed_df = pd.DataFrame()

    # Map class to an integer
    processed_df['price (label)'] = raw_df['quality']

    # All features are numerical and require no further processing
    for col in raw_df.columns:
        if col != 'quality':
            processed_df[col] = raw_df[col].round(decimals=4)

    processed_df.to_csv(processed_file, header=True, index=False)
    return processed_df.astype('float64')
