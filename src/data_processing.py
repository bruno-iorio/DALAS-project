import numpy as np 
import pandas as pd 
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")



def chunking(text):
    return text.split(".")

def encode(text):
    return model.encode(text)

def encode_list(list_of_text):
    other = list_of_text
    for i in range(len(list_of_text)):
        other[i] = encode(list_of_text[i])
    return other

def text_process_dataframe(df,columns):
    df2 = df.copy()
    for col in columns:
        list_to_add = []
        new_name = col + "_encoded"
        for i in tqdm(range(len(df))):
            text = df[col][i]
            if isinstance(text,str):
                chunked = chunking(text.lower())
                list_of_encoded = encode_list(chunked)
                list_to_add.append(list_of_encoded)
            else:
                list_to_add.append(np.nan)
        df2[new_name] = list_to_add 
    return df2
                






