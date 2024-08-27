
import numpy as np
import pandas as pd

# 
import text_load
import text_prepare

df_train = text_load.LoadData(source="train - train.csv")
print('clearing train')
df_train = text_prepare.clean_pd_data(df_train, 'comment_text')

df_test = text_load.LoadData(source="test - test.csv")
print('clearing test')
df_test = text_prepare.clean_pd_data(df_test, 'comment_text')
