import numpy as np 
import pandas as pd
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")



def chunking(text):
    return text.split(".")

def encode(text):
    return model.encode(text)

def encode(list_of_text):
    
