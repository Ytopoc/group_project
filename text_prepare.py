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
    print('data size ', pd_size)
    for index in range(pd_size):
        try:
            pd_data[column_name][index] = clean_text(pd_data[column_name][index])
        except:
            print('index = ', index)
            print('data = ', pd_data[column_name][index])
            print('id = ', pd_data['id'][index])
        if (index % 100)==0:
            prog = index/pd_size*100
            print(f"\r{prog:.2f} ", end="")
    return pd_data

def tokenize_and_encode(tokenizer, comments, labels, max_length=128):
    # Initialize empty lists to store tokenized inputs and attention masks
    input_ids = []
    attention_masks = []

    # Iterate through each comment in the 'comments' list
    for comment in comments:

        # Tokenize and encode the comment using the BERT tokenizer
        encoded_dict = tokenizer.encode_plus(
            comment,

            # Add special tokens like [CLS] and [SEP]
            add_special_tokens=True,

            # Truncate or pad the comment to 'max_length'
            max_length=max_length,

            # Pad the comment to 'max_length' with zeros if needed
            pad_to_max_length=True,

            # Return attention mask to mask padded tokens
            return_attention_mask=True,

            # Return PyTorch tensors
            return_tensors='pt'
        )

        # Append the tokenized input and attention mask to their respective lists
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Concatenate the tokenized inputs and attention masks into tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # Convert the labels to a PyTorch tensor with the data type float32
    labels = torch.tensor(labels, dtype=torch.float32)

    # Return the tokenized inputs, attention masks, and labels as PyTorch tensors
    return input_ids, attention_masks, labels
