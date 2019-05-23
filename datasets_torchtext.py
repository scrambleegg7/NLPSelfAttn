
from glob import glob
from time import time

from nltk.tokenize import RegexpTokenizer
from collections import defaultdict

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

import pandas as pd  

import torchtext 
import spacy
from bs4 import BeautifulSoup
import dill 

from torchtext.data import Example
from torchtext.datasets import language_modeling
from torchtext.data import Field, BucketIterator, TabularDataset, ReversibleField, Dataset
from torchtext.vocab import GloVe
from sklearn.model_selection import train_test_split

class BatchGenerator:
    def __init__(self, dl, x_field, y_field):
        self.dl, self.x_field, self.y_field = dl, x_field, y_field
        
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        for batch in self.dl:
            X = getattr(batch, self.x_field)
            y = getattr(batch, self.y_field)
            yield (X,y)


def main():


    data_dir = "/home/donchan/Documents/DATA/IMDB/aclImdb"
    #imdb = IMDBTextDataset(data_dir)

    df = pd.read_csv(os.path.join(data_dir,"imdb_data_train.csv"))
    print("pandas df size", df.shape)

    start_t = time()

    TEXT = Field(sequential=True, tokenize="spacy", lower=True)
    LABEL = Field(sequential=False)
    data_fields = [('Review', TEXT), ('Sensitivity', LABEL)]

    train, val = TabularDataset.splits(path=data_dir, train='train.csv', 
        validation='val.csv', format='csv', fields=data_fields)

    TEXT.build_vocab(train, vectors="glove.6B.100d", min_freq=2 )
    LABEL.build_vocab(train)

    print("time to build textdataset", time() - start_t)

    device = torch.device('cuda')
    train_iter , val_iter = BucketIterator.splits(
                                (train, val), 
                            batch_size=32, device=device, #sort_key=lambda x:len(x.text),
                            #sort_within_batch=True, 
                            repeat=False)

    batch = next( iter( train_iter ) )
    print(batch.Review)
    print(batch.Sensitivity)
    print(batch.dataset.fields)

    train_batch_it = BatchGenerator(train_iter, 'Review', 'Sensitivity')

    for batch in train_batch_it:
        text_data, label = batch

    print(text_data.shape)
    h,w = text_data.shape
    for i in range(w):
        text_id = text_data[:,i]
        
        string_list = []
        for j in range(h):
            str_ = TEXT.vocab.itos[ text_id[j] ]
            string_list.append( str_ )
    print(text_id)
    print(string_list)

if __name__ == "__main__":
    main()