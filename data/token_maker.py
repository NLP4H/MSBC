import pandas as pd 
import transformers 
from transformers import BertModel, BertTokenizer
import torch
import pdb
import numpy as np


df = pd.read_csv('master_path/data/neurology_notes/processed_data/neurology_notes_processed.csv')

# ------------- Optional section to convert MRN to PatID ------------------
df_id = pd.read_csv('master_path/data/neurology_notes/raw_data/Identification.csv')
pat_codes = df_id[['Patient ID', 'Patient Code']]
pat_codes = pat_codes.rename({'Patient ID': 'patient_id', 'Patient Code': 'MRN'}, axis='columns')
df = df.merge(pat_codes, on="MRN", how="right")
# -------------------------------------------------------------------------

df = pd.read_csv("master_path/results/task1/edss_19_results.csv")
# If the tokenizer uses a single vocabulary file
tokenizer = BertTokenizer.from_pretrained('master_path/models/base_blue_bert_pt/vocab.txt')
df['text'] = df.text.fillna('')
#replace new lines
df = df.replace({r'\s+$': '', r'^\s+': ''}, regex = True).replace(r'\n', ' ', regex = True)
# generates tokens 
token_list = df.apply(lambda x: tokenizer.encode(x['text'], add_special_tokens=True), axis = 1) 
#join on tabs

token_list = token_list.astype(str)
df['tokenized_text'] = token_list

df.to_csv("master_path/results/task1/edss_19_results_tokenized.csv")
df_train = df[df['split'] == 'unlabeled']

from sklearn.model_selection import train_test_split

df_train, df_val = train_test_split(df[df['split'] == 'unlabeled'], test_size=0.3)

df_train.to_csv("master_path/data/snorkel/train.csv")
df_val.to_csv("master_path/data/snorkel/val.csv")


df_test = df[df['split'] == 'test' & df['split'] == "val" & df['split'] == "train"]
df_test.to_csv("master_path/data/snorkel/test.csv")