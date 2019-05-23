#from EncRNN import EncoderRNN
#from attn import AttnClassifier

from glob import glob
from time import time

from nltk.tokenize import RegexpTokenizer
from collections import defaultdict

import torch
import torch.nn as nn   
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F

import tqdm
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

import pandas as pd  

import torchtext 
import spacy
from bs4 import BeautifulSoup
import dill 

from torchtext import vocab
from torchtext.data import Example
from torchtext.datasets import language_modeling
from torchtext.data import Field, BucketIterator, TabularDataset, ReversibleField, Dataset
from torchtext.vocab import GloVe
from sklearn.model_selection import train_test_split
from itertools import chain

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from SimpleLSTM import SimpleLSTMBaseline, SimpleBiLSTMBaseline

batch_size = 16

nlp = spacy.load('en',disable=['parser', 'tagger', 'ner'])


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = text.replace('x', '')
#    text = re.sub(r'\W+', '', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text



def tokenizer1(s): 
    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    #nltk_stopwords = nltk.corpus.stopwords.words('english')
    return [w.text.lower() for w in nlp(tweet_clean(s)) if not w.is_stop]

def tweet_clean(text):
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text) # remove non alphanumeric character
    text = re.sub(r'https?:/\/\S+', ' ', text) # remove links
    return text.strip()

def tokenizer2(s):
    
    tokenizer = RegexpTokenizer(r'\w+')     
    stop_words = stopwords.words('english')

    reg_token_words = tokenizer.tokenize(s)
    reg_token_words = [w.lower() for w in reg_token_words  if w.isalpha() ]

    words_filtered = reg_token_words[:] # creating a copy of the words list
    for word in reg_token_words:
        if word in stop_words:        
            words_filtered.remove(word)
        elif word in ["br"]:
            words_filtered.remove(word)

    return words_filtered

def build_csv():

    data_dir = "/home/donchan/Documents/DATA/jigsaw"

    df = pd.read_csv(os.path.join(data_dir,"train.csv"))
    print("pandas df size", df.shape)
    print(df.head(2))

    df =df[:10000].copy()

    df["comment_text"] = df["comment_text"].apply(clean_text)
    df["comment_text"] = df["comment_text"].str.replace('\d+', '')

    print("- "*20)
    train, val = train_test_split(df, test_size=0.3)
    train.to_csv( os.path.join(data_dir,"traindf.csv"), index=False )
    val.to_csv( os.path.join(data_dir,"valdf.csv"), index=False )


def dataset():

    build_csv()

    data_dir = "/home/donchan/Documents/DATA/jigsaw"

    start_t = time()
    
    vec = vocab.Vectors('glove.6B.100d.txt', '/home/donchan/Documents/DATA/glove_embedding/')

    TEXT = Field(sequential=True, tokenize=tokenizer2, lower=True)
    LABEL = Field(sequential=False, use_vocab=False)

    datafields = [("id", None), # we won't be needing the id, so we pass in None as the field
                 ("comment_text", TEXT), ("toxic", LABEL),
                 ("severe_toxic", LABEL), 
                 ("obscene", LABEL), ("threat", LABEL), ("insult", LABEL),
                 ("identity_hate", LABEL)]

    train, val = TabularDataset.splits(path=data_dir, train='traindf.csv', 
        validation='valdf.csv', format='csv', skip_header=True, fields=datafields)

    print("train val length", len(train), len(val))
    #print( train[0].comment_text )
    #print( train[0].toxic, train[0].severe_toxic, train[0].threat, train[0].insult, train[0].identity_hate  )

    TEXT.build_vocab(train, val, vectors=vec, min_freq=2 )
    #LABEL.build_vocab(train, val)

    print("time to build vocab", (time() - start_t))
    print("length of vocaburary", len(TEXT.vocab), TEXT.vocab.vectors.shape )

    print("- "*20 )
    print("* most common words.")
    print( TEXT.vocab.freqs.most_common(20) )

    return train, val, TEXT, LABEL


def main():

    train, val, TEXT, LABEL = dataset()

if __name__ == "__main__":
    main()