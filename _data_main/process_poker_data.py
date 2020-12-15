import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

def load_poker_data():

    raw_data_file = os.path.join(os.path.dirname(__file__), 'poker_raw.csv')
    processed_file = os.path.join(os.path.dirname(__file__), 'poker_processed.csv')

    ##### Poker Data Processing
    raw_df = pd.read_csv(raw_data_file)
    processed_df = pd.DataFrame()

    # All features are numerical and require no further processing
    processed_df['class (label)'] = raw_df['class']
    for col in raw_df.columns:
        if col != 'class':
            processed_df[col] = raw_df[col]
    processed_df = processed_df.astype('int64')

    processed_df.to_csv(processed_file, header=True, index=False)
    return processed_df
