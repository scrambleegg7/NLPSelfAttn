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
from torchtext.data import Field, BucketIterator, TabularDataset, ReversibleField, Dataset, Iterator
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

class BatchWrapper:
    def __init__(self, dl, x_var, y_vars):
        self.dl, self.x_var, self.y_vars = dl, x_var, y_vars # we pass in the list of attributes for x and y
    
    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x_var) # we assume only one input in this wrapper
            
            if self.y_vars is not None: # we will concatenate y into a single tensor
                y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1).float()
            else:
                y = torch.zeros((1))

            yield (x, y)
    
    def __len__(self):
        return len(self.dl)



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

    print("- "*20)
    train, val = train_test_split(df, test_size=0.2)
    train.to_csv( os.path.join(data_dir,"traindf.csv"), index=False )
    val.to_csv( os.path.join(data_dir,"valdf.csv"), index=False )


def main():


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

    tst_datafields = [("id", None), # we won't be needing the id, so we pass in None as the field
                    ("comment_text", TEXT)
    ]
    tst = TabularDataset(
            path=os.path.join(data_dir,"test.csv"), # the file path
            format='csv',
            skip_header=True, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
            fields=tst_datafields)
    print("test length", len(tst))
    print("time to build vocab", (time() - start_t))
    print("length of vocaburary", len(TEXT.vocab), TEXT.vocab.vectors.shape )

    print("- "*20 )
    print("* most common words. from train and val data.")
    print( TEXT.vocab.freqs.most_common(20) )

    test_iter = Iterator(tst, batch_size=batch_size, device=torch.device("cuda"), sort=False, sort_within_batch=False, repeat=False)
    test_dl = BatchWrapper(test_iter, "comment_text", None)

    em_sz = 100
    nh = 500


    if torch.cuda.is_available():
        print("cuda")
    else:
        print("cpu")

    model_file = os.path.join(data_dir, "jigsaw_model_7978.pkl")
    model = SimpleBiLSTMBaseline( hidden_dim=nh,emb_dim=em_sz,len_TEXT_vocab=len(TEXT.vocab), v_vec=TEXT.vocab.vectors )

    if os.path.isfile( model_file  ):
        print("model file found.", )
        model.load_state_dict( torch.load( model_file, map_location='cuda:0' ) )
    model.cuda()

    model.eval() # turn on training mode
    
    test_preds = []
    for idx, (x, y) in enumerate( tqdm.tqdm( test_dl ) ): # thanks to our wrapper, we can intuitively iterate over our data!

        preds = model(x)
        m = nn.Sigmoid()
        logits = m(preds)

        logits = logits.cpu().data.numpy()

        if idx % 500 == 0:
            print("logits",logits)
            print(logits.shape)

        test_preds.append(logits)


    test_preds = np.vstack(test_preds)
    print("final preds shape", test_preds.shape)

    df = pd.read_csv( os.path.join( data_dir,  "test.csv" ) )
    print("dataframe shape", df.shape)
    for i, col in enumerate(["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]):
        df[col] = test_preds[:, i]

    df.to_csv( os.path.join(data_dir,"test_final_prob.csv"), index=False )

if __name__ == "__main__":
    main()