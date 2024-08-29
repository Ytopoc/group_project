import torch
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset

#to avoid warnings
import warnings
warnings.filterwarnings('ignore')

st.markdown("""
    <style>
        ul {
            column-count: 2;
        }

        ul li {
            font-size: 16px;
            text-align: left;
            list-style: none;
            color: #eee;
        }
                
                    ul li span {
                        padding: 0px 10px;
                    }

                    ul li .round {
                        width: 6px;
                        height: 6px;
                        border-radius: 50%;
                        background: #eee;
                        float: left;
                        margin-top: 7px;
                        padding: 6px 6px;
                    }
                
                    ul li.active-danger {
                        color: #D44942;
                    }
                
                    ul li.active-danger .round {
                        background: #D44942;
                    }
                
                    ul li.active-success {
                        color: #46D667;
                    }
                
                    ul li.active-success .round {
                        background: #46D667;
                    }
                </style>""", 
            unsafe_allow_html=True)

# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

PATH = 'saved_model/'
Bert_Tokenizer = BertTokenizer.from_pretrained(PATH, local_files_only=True)
Bert_Model = BertForSequenceClassification.from_pretrained(PATH, local_files_only=True)

def predict_user_input(input_text, model=Bert_Model, tokenizer=Bert_Tokenizer, device=device):
    user_input = [input_text]

    user_encodings = tokenizer(
        user_input, truncation=True, padding=True, return_tensors="pt")

    user_dataset = TensorDataset(
        user_encodings['input_ids'], user_encodings['attention_mask'])

    user_loader = DataLoader(user_dataset, batch_size=1, shuffle=False)

    model.eval()
    with torch.no_grad():
        for batch in user_loader:
            input_ids, attention_mask = [t.to(device) for t in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.sigmoid(logits)

    predicted_labels = (predictions.cpu().numpy() > 0.5).astype(int)
    labels_list = ['toxic', 'severe_toxic', 'obscene',
                   'threat', 'insult', 'identity_hate']
    result = dict(zip(labels_list, predicted_labels[0]))
    return result

def show_result(result):

    response = {
        'toxic': 'Toxic',
        'severe_toxic': 'Severe toxic',
        'obscene': 'Obscene',
        'threat': 'Threat',
        'insult': 'Insult',
        'identity_hate': 'Identity hate'
    }

    items = ''
    for key, name in response.items():
        active = ''
        if result and result[key]:
            active = 'class="active-danger"'
            
        items += f"""<li """+active+"""><div class='round'></div><span>"""+ name +"""</span></li>"""

    non_zeros = dict(filter(lambda kv: kv[1] != 0, result.items()))

    success = ' class="active-success"' if result and not non_zeros else ''

    st.html(
        f"""
            <div class='card'>
                <ul>
                    <li{success}><div class='round'></div><span>Passed</span></li>
                </ul>
                <hr/>
                <ul>
                    {items}  
                </ul>
            </div>
        """
    )

    

def main():
    st.header('Toxic Comments Classification form')

    with st.form('addition'):
        
        def clear_form():
            st.session_state.comment_input = ""

        comment = st.text_area('Enter your comment bellow:', key='comment_input')
        submit_form = st.form_submit_button('Check comment')

        if submit_form:
            show_result(
                predict_user_input(input_text=comment)
            )

    st.button('Clear form', on_click=clear_form)

if __name__ == '__main__':
    main()