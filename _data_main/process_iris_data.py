import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

def load_iris_data():

    raw_data_file = os.path.join(os.path.dirname(__file__), 'iris_raw.csv')
    processed_file = os.path.join(os.path.dirname(__file__), 'iris_processed.csv')

    ##### Iris Data Processing
    raw_df = pd.read_csv(raw_data_file)
    processed_df = pd.DataFrame()

    # Map class to an integer
    class_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    processed_df['class (label)'] = raw_df['class'].map(class_map)

    # All features are numerical and require no further processing
    for col in raw_df.columns:
        if col != 'class':
            processed_df[col] = raw_df[col]

    processed_df.to_csv(processed_file, header=True, index=False)
    return processed_df.astype('float64')
