
import numpy as np
import pandas as pd

# 
sources_local = ('train - train.csv', 'test - test.csv')
sources_remote = (
            'https://docs.google.com/spreadsheets/d/1I-WycB-n8VvkKrmEvuvA4n-ctcMYyH9o58T783Byt2k/export?format=csv',
            'https://docs.google.com/spreadsheets/d/17J2k4Cz8FwTGEyQ2mwLLKbFgqbgK9D2a5zfQiuJBt6M/export?format=csv')

#   def LoadData(source)
#   завантаження даних у DataFrame
import text_load

#   def clean_pd_data(pd_data, column_name)
#   приведення рядків у нижній формат та видалення знаків пунктуації
#   def tokenize_and_encode(tokenizer, comments, labels, max_length=128)
import text_prepare

#import text_tokenize

df_train = text_load.LoadData(source=sources_local[0])
print('clearing train')
# df_train = text_prepare.clean_pd_data(df_train, 'comment_text')

df_test = text_load.LoadData(source=sources_local[1])
print('clearing test')
df_test = text_prepare.clean_pd_data(df_test, 'comment_text')
