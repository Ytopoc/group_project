#import numpy as np

import pandas as pd

#to avoid warnings
import warnings

warnings.filterwarnings('ignore')

def LoadData(source = "my.csv"):
    pd_data = pd.read_csv(source)
    return pd_data

def LoadTrainData(source = 'local'):
    print('train data', end=" ")
    if source == 'local':
        pd_data = pd.read_csv('train - train.csv')
    else:
        pd_data = pd.read_csv('https://docs.google.com/spreadsheets/d/1I-WycB-n8VvkKrmEvuvA4n-ctcMYyH9o58T783Byt2k/export?format=csv')
    print(', done')
    return pd_data

def LoadTestData(source = 'local'):
    print('test data', end=" ")
    if source == 'local':
        pd_data = pd.read_csv('test - test.csv')
    else:
        pd_data = pd.read_csv('https://docs.google.com/spreadsheets/d/17J2k4Cz8FwTGEyQ2mwLLKbFgqbgK9D2a5zfQiuJBt6M/export?format=csv')
    print(', done')
    return pd_data

