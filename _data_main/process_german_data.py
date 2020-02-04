# from recourse.path import *
import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

def load_german_data():

  # input vars
  data_name = 'german'
  raw_data_file = os.path.join(os.path.dirname(__file__), 'german_raw.csv')
  processed_file = os.path.join(os.path.dirname(__file__), 'german_processed.csv')

  ##### German Data Processing
  raw_df = pd.read_csv(raw_data_file) # , index_col = 0)
  processed_df = pd.DataFrame()

  processed_df['GoodCustomer (label)'] = raw_df['GoodCustomer']
  processed_df['GoodCustomer (label)'] = (processed_df['GoodCustomer (label)'] + 1) / 2
  processed_df.loc[raw_df['Gender'] == 'Male', 'Sex'] = 1
  processed_df.loc[raw_df['Gender'] == 'Female', 'Sex'] = 0
  processed_df['Age'] = raw_df['Age']
  processed_df['Credit'] = raw_df['Credit']
  processed_df['LoanDuration'] = raw_df['LoanDuration']

  # # order important, more balance can overwrite less balance!
  # processed_df.loc[raw_df['CheckingAccountBalance_geq_0'] == 1, 'CheckingAccountBalance'] = 2
  # processed_df.loc[raw_df['CheckingAccountBalance_geq_200'] == 1, 'CheckingAccountBalance'] = 3
  # processed_df = processed_df.fillna(1) # all other categories...

  # # order important, more balance can overwrite less balance!
  # processed_df.loc[raw_df['SavingsAccountBalance_geq_100'] == 1, 'SavingsAccountBalance'] = 2
  # processed_df.loc[raw_df['SavingsAccountBalance_geq_500'] == 1, 'SavingsAccountBalance'] = 3
  # processed_df = processed_df.fillna(1) # all other categories...

  # # 2: owns house, 1: rents house, 0: neither
  # processed_df.loc[raw_df['OwnsHouse'] == 1, 'HousingStatus'] = 3
  # processed_df.loc[raw_df['RentsHouse'] == 1, 'HousingStatus'] = 2
  # processed_df = processed_df.fillna(1) # all other categories...

  # Save to CSV
  processed_df = processed_df + 0 # convert boolean values to numeric
  processed_df = processed_df.reset_index(drop = True)
  processed_df = processed_df.dropna() # drop all rows that include NAN (some exist in isMarried column, possibly elsewhere as well)
  processed_df.to_csv(processed_file, header = True, index = False)
  assert(processed_df.shape[0] == 1000)

  return processed_df.astype('float64')
