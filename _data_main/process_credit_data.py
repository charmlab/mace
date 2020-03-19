# from recourse.path import *
import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

def load_credit_data():

  # input vars
  data_name = 'credit'
  raw_data_file = os.path.join(os.path.dirname(__file__), 'credit_raw.csv')
  processed_file = os.path.join(os.path.dirname(__file__), 'credit_processed.csv')

  ##### Credit Data Processing
  raw_df = pd.read_csv(raw_data_file, index_col = 0)
  processed_df = pd.DataFrame()

  # convert NTD to USD using spot rate in 09-2005
  NTD_to_USD = 32.75 # see https://www.poundsterlinglive.com/bank-of-england-spot/historical-spot-exchange-rates/usd/USD-to-TWD-2005
  monetary_features = list(filter(lambda x: ('BILL_AMT' in x) or ('PAY_AMT' in x) or ('LIMIT_BAL' in x), raw_df.columns))
  raw_df[monetary_features] = raw_df[monetary_features].applymap(lambda x: x / NTD_to_USD).round(-1).astype(int)

  # outcome variable in first column
  processed_df['NoDefaultNextMonth (label)'] = 1 - raw_df['default payment next month (label)']

  # Gender (old; male = 1, female = 2) --> (new; male = 0, female = 1)
  # <Removed by Berk> processed_df['Female'] = raw_df['SEX'] == 2
  processed_df.loc[raw_df['SEX'] == 1, 'isMale'] = True
  processed_df.loc[raw_df['SEX'] == 2, 'isMale'] = False

  # Married (old; married = 1; single = 2; other = 3) --> (new; married = 1; single = 2; other = 3)
  # <Removed by Amir> processed_df['Married'] = raw_df['MARRIAGE'] == 1
  # <Removed by Amir> processed_df['Single'] = raw_df['MARRIAGE'] == 2
  processed_df.loc[raw_df.MARRIAGE == 1, 'isMarried'] = True # married (use T/F, but not 1/0, so that some values become NAN and can be dropped later!)
  processed_df.loc[raw_df.MARRIAGE == 2, 'isMarried'] = False # single (use T/F, but not 1/0, so that some values become NAN and can be dropped later!)
  # <Set to NAN by Amir> processed_df.loc[raw_df.MARRIAGE == 0, 'isMarried'] = 3 # other
  # <Set to NAN by Amir> processed_df.loc[raw_df.MARRIAGE == 3, 'isMarried'] = 3 # other

  # Age
  # <Removed by Amir> processed_df['Age_lt_25'] = raw_df['AGE'] < 25
  # <Removed by Amir> processed_df['Age_in_25_to_40'] = raw_df['AGE'].between(25, 40, inclusive = True)
  # <Removed by Amir> processed_df['Age_in_40_to_59'] = raw_df['AGE'].between(40, 59, inclusive = True)
  # <Removed by Amir> processed_df['Age_geq_60'] = raw_df['AGE'] >= 60
  processed_df.loc[raw_df['AGE'] < 25, 'AgeGroup'] = 1
  processed_df.loc[raw_df['AGE'].between(25, 40, inclusive = True), 'AgeGroup'] = 2
  processed_df.loc[raw_df['AGE'].between(40, 59, inclusive = True), 'AgeGroup'] = 3
  processed_df.loc[raw_df['AGE'] >= 60, 'AgeGroup'] = 4

  # EducationLevel (currently, 1 = graduate school; 2 = university; 3 = high school; 4 = others)
  processed_df['EducationLevel'] = 1
  processed_df['EducationLevel'][raw_df['EDUCATION'] == 3] = 2 # HS
  processed_df['EducationLevel'][raw_df['EDUCATION'] == 2] = 3 # University
  processed_df['EducationLevel'][raw_df['EDUCATION'] == 1] = 4 # Graduate


  # Process Bill Related Variables
  pay_columns = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
  bill_columns = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']

  #processed_df['LastBillAmount'] = np.maximum(raw_df['BILL_AMT1'], 0)
  processed_df['MaxBillAmountOverLast6Months'] = np.maximum(raw_df[bill_columns].max(axis = 1), 0)
  processed_df['MaxPaymentAmountOverLast6Months'] = np.maximum(raw_df[pay_columns].max(axis = 1), 0)
  processed_df['MonthsWithZeroBalanceOverLast6Months'] = np.sum(np.greater(raw_df[pay_columns].values, raw_df[bill_columns].values), axis = 1)
  processed_df['MonthsWithLowSpendingOverLast6Months'] = np.sum(raw_df[bill_columns].div(raw_df['LIMIT_BAL'], axis = 0) < 0.20, axis = 1)
  processed_df['MonthsWithHighSpendingOverLast6Months'] = np.sum(raw_df[bill_columns].div(raw_df['LIMIT_BAL'], axis = 0) > 0.80, axis = 1)
  processed_df['MostRecentBillAmount'] = np.maximum(raw_df[bill_columns[0]], 0)
  processed_df['MostRecentPaymentAmount'] = np.maximum(raw_df[pay_columns[0]], 0)

  # Credit History
  # PAY_M' = months since last payment (as recorded last month)
  # PAY_6 =  months since last payment (as recorded 6 months ago)
  # PAY_M = -1 if paid duly in month M
  # PAY_M = -2 if customer was issued refund M
  raw_df = raw_df.rename(columns = {'PAY_0': 'MonthsOverdue_1',
                                    'PAY_2': 'MonthsOverdue_2',
                                    'PAY_3': 'MonthsOverdue_3',
                                    'PAY_4': 'MonthsOverdue_4',
                                    'PAY_5': 'MonthsOverdue_5',
                                    'PAY_6': 'MonthsOverdue_6'})

  overdue = ['MonthsOverdue_%d' % j for j in range(1, 7)]
  raw_df[overdue] = raw_df[overdue].replace(to_replace = [-2, -1], value = [0, 0])
  overdue_history = raw_df[overdue].to_numpy() > 0
  payment_history = np.logical_not(overdue_history)

  def count_zero_streaks(a):
      #adapted from zero_runs function of https://stackoverflow.com/a/24892274/568249
      iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
      absdiff = np.abs(np.diff(iszero))
      runs = np.where(absdiff == 1)[0].reshape(-1, 2)
      n_streaks = runs.shape[0]
      #streak_lengths = np.sum(runs[:,1] - runs[:,0])
      return n_streaks


  overdue_counts = np.repeat(np.nan, len(raw_df))
  n_overdue_months = np.sum(overdue_history > 0, axis = 1)
  overdue_counts[n_overdue_months == 0] = 0 # count_zero_streaks doesn't work for edge cases
  overdue_counts[n_overdue_months == 6] = 1
  for k in range(1, len(overdue)):
      idx = n_overdue_months == k
      overdue_counts[idx] = [count_zero_streaks(a) for a in payment_history[idx, :]]

  overdue_counts = overdue_counts.astype(np.int_)
  processed_df['TotalOverdueCounts'] = overdue_counts
  processed_df['TotalMonthsOverdue'] = raw_df[overdue].sum(axis = 1)
  processed_df['HasHistoryOfOverduePayments'] = raw_df[overdue].sum(axis = 1) > 0


  # Save to CSV
  processed_df = processed_df + 0 # convert boolean values to numeric
  processed_df = processed_df.reset_index(drop = True)
  processed_df = processed_df.dropna() # drop all rows that include NAN (some exist in isMarried column, possibly elsewhere as well)
  processed_df.to_csv(processed_file, header = True, index = False)
  assert(processed_df.shape[0] == 29623)

  return processed_df.astype('float64')
