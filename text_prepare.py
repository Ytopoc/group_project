import re
import string

def clean_text(text):
    # Перетворюємо текст у нижній регістр
    text = text.lower()
    
    # Видаляємо URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Видаляємо HTML-теги
    text = re.sub(r'<.*?>', '', text)
    
    # Видаляємо числові дані
    text = re.sub(r'\d+', '', text)
    
    # Видаляємо пунктуацію
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Видаляємо зайві пробіли
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)

    return text

def clean_pd_data(pd_data, column_name):
    pd_size = len( pd_data[column_name] )
    error_idx = []
    print('data size ', pd_size)
    for index in range(pd_size):
        try:
            pd_data[column_name][index] = clean_text(pd_data[column_name][index])
        except:
            print('error in cleaning data:')
            print(' ==> index = ', index)
            print(' ==> data = ', pd_data[column_name][index])
            print(' ==> id = ', pd_data['id'][index])
            error_idx.append(index)
        if (index % 100)==0:
            prog = index/pd_size*100
            print(f"\r{prog:.2f} % ", end="")
    print("\r100% of data cleared ...")
    if len(error_idx)>0:
        print('removing data with errors in DataFrame:')
        while len(error_idx)>0:
            pd_data.drop(index = error_idx[len(error_idx)-1])
            error_idx.pop()
    return pd_data

