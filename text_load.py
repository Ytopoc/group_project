import pandas as pd

#to avoid warnings
import warnings

warnings.filterwarnings('ignore')

def LoadData(source = "my.csv"):
    pd_data = pd.read_csv(source)
    return pd_data
