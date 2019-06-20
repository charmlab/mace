import os,sys
import numpy as np
import pandas as pd
import urllib.request
import fair_utils_data as ut
from random import seed, shuffle

SEED = 1122334455
seed(SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)

"""
    The adult dataset can be obtained from: http://archive.ics.uci.edu/ml/datasets/Adult
    The code will look for the data files (adult.data, adult.test) in the present directory, if they are not found, it will download them from UCI archive.
"""

def check_data_file(file_name):
    this_files_directory = os.path.dirname(os.path.realpath(__file__))
    files_in_directory = os.listdir(this_files_directory) # get the current directory listing

    print(f'Looking for file {file_name} in the {this_files_directory} directory..')

    if file_name not in files_in_directory:
        full_file_name = os.path.join(this_files_directory, file_name)
        print("'%s' not found! Downloading from UCI Archive..." % file_name)
        addr = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/%s" % file_name
        response = urllib.request.urlopen(addr)
        data = response.read()
        fileOut = open(full_file_name, "wb")
        fileOut.write(data)
        fileOut.close()
        print("'%s' download and saved locally.." % full_file_name)
    else:
        print("File found in current directory..")

    print()
    return


def load_adult_data(load_data_size=None):

    """
        if load_data_size is set to None (or if no argument is provided), then we load and return the whole data
        if it is a number, say 10000, then we will return randomly selected 10K examples
    """

    attrs = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country'] # all attributes
    int_attrs = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week'] # attributes with integer values -- the rest are categorical
    sensitive_attrs = ['sex'] # the fairness constraints will be used for this feature
    attrs_to_ignore = ['sex', 'race' ,'fnlwgt'] # sex and race are sensitive feature so we will not use them in classification, we will not consider fnlwght for classification since its computed externally and it highly predictive for the class (for details, see documentation of the adult data)
    attrs_for_classification = set(attrs) - set(attrs_to_ignore)

    # adult data comes in two different files, one for training and one for testing, however, we will combine data from both the files
    data_files = ["adult.data", "adult.test"]


    X = []
    y = []
    x_control = {}

    attrs_to_vals = {} # will store the values for each attribute for all users
    for k in attrs:
        if k in sensitive_attrs:
            x_control[k] = []
        elif k in attrs_to_ignore:
            pass
        else:
            attrs_to_vals[k] = []

    for f in data_files:
        check_data_file(f)

        for line in open(f):
            line = line.strip()
            if line == "": continue # skip empty lines
            line = line.split(", ")
            if len(line) != 15 or "?" in line: # if a line has missing attributes, ignore it
                continue

            class_label = line[-1]
            if class_label in ["<=50K.", "<=50K"]:
                class_label = -1
            elif class_label in [">50K.", ">50K"]:
                class_label = +1
            else:
                raise Exception("Invalid class label value")

            y.append(class_label)

            for i in range(0,len(line)-1):
                attr_name = attrs[i]
                attr_val = line[i]
                # reducing dimensionality of some very sparse features
                if attr_name == "native_country":
                    if attr_val!="United-States":
                        attr_val = "Non-United-Stated"
                elif attr_name == "education":
                    if attr_val in ["Preschool", "1st-4th", "5th-6th", "7th-8th"]:
                        attr_val = "prim-middle-school"
                    elif attr_val in ["9th", "10th", "11th", "12th"]:
                        attr_val = "high-school"

                if attr_name in sensitive_attrs:
                    x_control[attr_name].append(attr_val)
                elif attr_name in attrs_to_ignore:
                    pass
                else:
                    attrs_to_vals[attr_name].append(attr_val)


    def convert_attrs_to_ints(d): # discretize the string attributes
        for attr_name, attr_vals in d.items():
            if attr_name in int_attrs:
                continue
            else:
                uniq_vals = sorted(list(set(attr_vals))) # get unique values

                # compute integer codes for the unique values
                val_dict = {}
                for i in range(0,len(uniq_vals)):
                    val_dict[uniq_vals[i]] = i

                # replace the values with their integer encoding
                for i in range(0,len(attr_vals)):
                    attr_vals[i] = val_dict[attr_vals[i]]
                d[attr_name] = attr_vals


    # convert the discrete values to their integer representations
    convert_attrs_to_ints(x_control)
    convert_attrs_to_ints(attrs_to_vals)


    # if the integer vals are not binary, we need to get one-hot encoding for them
    for attr_name in attrs_for_classification:
        attr_vals = attrs_to_vals[attr_name]
        if attr_name in int_attrs or attr_name == "native_country": # the way we encoded native country, its binary now so no need to apply one hot encoding on it
            X.append(attr_vals)

        else:
            attr_vals, index_dict = ut.get_one_hot_encoding(attr_vals)
            for inner_col in attr_vals.T:
                X.append(inner_col)


    # convert to numpy arrays for easy handline
    X = np.array(X, dtype=float).T
    y = np.array(y, dtype = float)
    for k, v in x_control.items(): x_control[k] = np.array(v, dtype=float)

    # shuffle the data
    perm = list(range(0,len(y))) # shuffle the data before creating each fold
    shuffle(perm)
    X = X[perm]
    y = y[perm]
    for k in x_control.keys():
        x_control[k] = x_control[k][perm]

    # see if we need to subsample the data
    if load_data_size is not None:
        print("Loading only %d examples from the data" % load_data_size)
        X = X[:load_data_size]
        y = y[:load_data_size]
        for k in x_control.keys():
            x_control[k] = x_control[k][:load_data_size]

    return X, y, x_control


def load_adult_data_new():

    attrs = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country'] # all attributes
    int_attrs = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week'] # attributes with integer values -- the rest are categorical
    sensitive_attrs = ['sex'] # the fairness constraints will be used for this feature
    attrs_to_ignore = ['sex', 'race' ,'fnlwgt'] # sex and race are sensitive feature so we will not use them in classification, we will not consider fnlwght for classification since its computed externally and it highly predictive for the class (for details, see documentation of the adult data)
    attrs_for_classification = set(attrs) - set(attrs_to_ignore)

    # adult data comes in two different files, one for training and one for testing, however, we will combine data from both the files
    this_files_directory = os.path.dirname(os.path.realpath(__file__))
    data_files = ["adult.data", "adult.test"]

    X = []
    y = []
    x_control = {}

    attrs_to_vals = {} # will store the values for each attribute for all users
    for k in attrs:
        if k in sensitive_attrs:
            x_control[k] = []
        elif k in attrs_to_ignore:
            pass
        else:
            attrs_to_vals[k] = []

    for file_name in data_files:

        check_data_file(file_name)
        full_file_name = os.path.join(this_files_directory, file_name)
        print(full_file_name)

        for line in open(full_file_name):
            line = line.strip()
            if line == "": continue # skip empty lines
            line = line.split(", ")
            if len(line) != 15 or "?" in line: # if a line has missing attributes, ignore it
                continue

            class_label = line[-1]
            if class_label in ["<=50K.", "<=50K"]:
                class_label = 0
            elif class_label in [">50K.", ">50K"]:
                class_label = +1
            else:
                raise Exception("Invalid class label value")

            y.append(class_label)

            for i in range(0,len(line)-1):
                attr_name = attrs[i]
                attr_val = line[i]
                # reducing dimensionality of some very sparse features
                if attr_name == "native_country":
                    if attr_val!="United-States":
                        attr_val = "Non-United-Stated"
                elif attr_name == "education":
                    if attr_val in ["Preschool", "1st-4th", "5th-6th", "7th-8th"]:
                        attr_val = "prim-middle-school"
                    elif attr_val in ["9th", "10th", "11th", "12th"]:
                        attr_val = "high-school"

                if attr_name in sensitive_attrs:
                    x_control[attr_name].append(attr_val)
                elif attr_name in attrs_to_ignore:
                    pass
                else:
                    attrs_to_vals[attr_name].append(attr_val)


    all_attrs_to_vals = attrs_to_vals
    all_attrs_to_vals['sex'] = x_control['sex']
    all_attrs_to_vals['label'] = y

    first_key = list(all_attrs_to_vals.keys())[0]
    for key in all_attrs_to_vals.keys():
      assert(len(all_attrs_to_vals[key]) == len(all_attrs_to_vals[first_key]))

    df = pd.DataFrame.from_dict(all_attrs_to_vals)

    processed_df = pd.DataFrame()

    processed_df['Label'] = df['label']

    processed_df.loc[df['sex'] == 'Male', 'Sex'] = 1
    processed_df.loc[df['sex'] == 'Female', 'Sex'] = 2

    processed_df['Age'] = df['age']

    processed_df.loc[df['native_country'] == 'United-States', 'NativeCountry'] = 1
    processed_df.loc[df['native_country'] == 'Non-United-Stated', 'NativeCountry'] = 2

    processed_df.loc[df['workclass'] == 'Federal-gov', 'WorkClass'] = 1
    processed_df.loc[df['workclass'] == 'Local-gov', 'WorkClass'] = 2
    processed_df.loc[df['workclass'] == 'Private', 'WorkClass'] = 3
    processed_df.loc[df['workclass'] == 'Self-emp-inc', 'WorkClass'] = 4
    processed_df.loc[df['workclass'] == 'Self-emp-not-inc', 'WorkClass'] = 5
    processed_df.loc[df['workclass'] == 'State-gov', 'WorkClass'] = 6
    processed_df.loc[df['workclass'] == 'Without-pay', 'WorkClass'] = 7

    processed_df['EducationNumber'] = df['education_num']

    processed_df.loc[df['education'] == 'prim-middle-school', 'EducationLevel'] = 1
    processed_df.loc[df['education'] == 'high-school', 'EducationLevel'] = 2
    processed_df.loc[df['education'] == 'HS-grad', 'EducationLevel'] = 3
    processed_df.loc[df['education'] == 'Some-college', 'EducationLevel'] = 4
    processed_df.loc[df['education'] == 'Bachelors', 'EducationLevel'] = 5
    processed_df.loc[df['education'] == 'Masters', 'EducationLevel'] = 6
    processed_df.loc[df['education'] == 'Doctorate', 'EducationLevel'] = 7
    processed_df.loc[df['education'] == 'Assoc-voc', 'EducationLevel'] = 8
    processed_df.loc[df['education'] == 'Assoc-acdm', 'EducationLevel'] = 9
    processed_df.loc[df['education'] == 'Prof-school', 'EducationLevel'] = 10

    processed_df.loc[df['marital_status'] == 'Divorced', 'MaritalStatus'] = 1
    processed_df.loc[df['marital_status'] == 'Married-AF-spouse', 'MaritalStatus'] = 2
    processed_df.loc[df['marital_status'] == 'Married-civ-spouse', 'MaritalStatus'] = 3
    processed_df.loc[df['marital_status'] == 'Married-spouse-absent', 'MaritalStatus'] = 4
    processed_df.loc[df['marital_status'] == 'Never-married', 'MaritalStatus'] = 5
    processed_df.loc[df['marital_status'] == 'Separated', 'MaritalStatus'] = 6
    processed_df.loc[df['marital_status'] == 'Widowed', 'MaritalStatus'] = 7

    processed_df.loc[df['occupation'] == 'Adm-clerical', 'Occupation'] = 1
    processed_df.loc[df['occupation'] == 'Armed-Forces', 'Occupation'] = 2
    processed_df.loc[df['occupation'] == 'Craft-repair', 'Occupation'] = 3
    processed_df.loc[df['occupation'] == 'Exec-managerial', 'Occupation'] = 4
    processed_df.loc[df['occupation'] == 'Farming-fishing', 'Occupation'] = 5
    processed_df.loc[df['occupation'] == 'Handlers-cleaners', 'Occupation'] = 6
    processed_df.loc[df['occupation'] == 'Machine-op-inspct', 'Occupation'] = 7
    processed_df.loc[df['occupation'] == 'Other-service', 'Occupation'] = 8
    processed_df.loc[df['occupation'] == 'Priv-house-serv', 'Occupation'] = 9
    processed_df.loc[df['occupation'] == 'Prof-specialty', 'Occupation'] = 10
    processed_df.loc[df['occupation'] == 'Protective-serv', 'Occupation'] = 11
    processed_df.loc[df['occupation'] == 'Sales', 'Occupation'] = 12
    processed_df.loc[df['occupation'] == 'Tech-support', 'Occupation'] = 13
    processed_df.loc[df['occupation'] == 'Transport-moving', 'Occupation'] = 14

    processed_df.loc[df['relationship'] == 'Husband', 'Relationship'] = 1
    processed_df.loc[df['relationship'] == 'Not-in-family', 'Relationship'] = 2
    processed_df.loc[df['relationship'] == 'Other-relative', 'Relationship'] = 3
    processed_df.loc[df['relationship'] == 'Own-child', 'Relationship'] = 4
    processed_df.loc[df['relationship'] == 'Unmarried', 'Relationship'] = 5
    processed_df.loc[df['relationship'] == 'Wife', 'Relationship'] = 6

    processed_df['CapitalGain'] = df['capital_gain']
    processed_df['CapitalLoss'] = df['capital_loss']
    processed_df['HoursPerWeek'] = df['hours_per_week']


    return processed_df.astype('float64')
