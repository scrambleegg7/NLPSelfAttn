
from EncRNN import EncoderRNN
from attn import AttnClassifier

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
from torchtext import vocab


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


batch_size = 16

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


def weights_init(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Embedding') == -1):
            nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

def train_model(epoch, encoder, classifier, train_iter, optimizer, log_interval=10):


    
    encoder.train()
    classifier.train()
    correct = 0
    for idx, batch in enumerate(train_iter):
        x, y = batch
        
        x = torch.t(x)
        y = y - 1

        if idx == 0:
            print("review shape", x.shape)
            print("label shape", y.shape)
        #(x, x_l), y = batch.text, batch.label - 1
        optimizer.zero_grad()
        encoder_outputs = encoder(x)
        output, attn = classifier(encoder_outputs)
        loss = F.nll_loss(output, y)
        
        loss.backward()
        optimizer.step()
    
        pred = output.data.max(1, keepdim=True)[1]

        print("output", output)
        print("real label", y)

        correct += pred.eq(y.data.view_as(pred)).cpu().sum()
        if idx % log_interval == 0:
            print('train epoch: {} [{}/{}], acc:{}, loss:{}'.format(
                epoch, idx*len(x), len(train_iter)*batch_size,
                correct/float(log_interval * len(x)),
                loss.data.item() ) )
            

            correct = 0
        
        if idx > 4:
            break

def main():


    data_dir = "/home/donchan/Documents/DATA/IMDB/aclImdb"
    #imdb = IMDBTextDataset(data_dir)

    df = pd.read_csv(os.path.join(data_dir,"imdb_data_train.csv"))
    uniq_sens = df.loc[:,"sensitivity"].unique()
    print("pandas df size", df.shape)
    print("unique sensitivity", uniq_sens   )
    print("length of unique sensitivity ", len(uniq_sens))

    start_t = time()

    TEXT = Field(sequential=True, tokenize="spacy", lower=True)
    LABEL = Field(sequential=False)
    data_fields = [('Review', TEXT), ('Sensitivity', LABEL)]

    train, val = TabularDataset.splits(path=data_dir, train='imdb_data_train.csv', 
        validation='imdb_data_test.csv', format='csv', fields=data_fields)

    vec = vocab.Vectors('glove.6B.100d.txt', '/home/donchan/Documents/DATA/glove_embedding/')

    TEXT.build_vocab(train, vectors=vec, min_freq=2 )
    LABEL.build_vocab(train)

    print("time to build textdataset", time() - start_t)
    print("total vocab length", len(TEXT.vocab))
    device = torch.device('cuda')
    train_iter , val_iter = BucketIterator.splits(
                                (train, val), 
                            batch_size=batch_size, device=device, #sort_key=lambda x:len(x.text),
                            #sort_within_batch=True, 
                            repeat=False)

    batch = next( iter( train_iter ) )
    print(batch.Review)
    print(batch.Sensitivity)
    print(batch.dataset.fields)

    train_batch_it = BatchGenerator(train_iter, 'Review', 'Sensitivity')

    # make model
    emb_dim = 100
    h_dim = 64

    c_num = len( uniq_sens )
    c_num = 10
    encoder = EncoderRNN(emb_dim, h_dim, len(TEXT.vocab), 
                         gpu=True, v_vec = TEXT.vocab.vectors)
    classifier = AttnClassifier(h_dim, c_num)    

    cuda = True
    if cuda:
        encoder.cuda()
        classifier.cuda()

    for m in encoder.modules():
            print(m.__class__.__name__)
            weights_init(m)

    for m in classifier.modules():
        print(m.__class__.__name__)
        weights_init(m)

    #optimum
    lr = 0.001
    optimizer = optim.Adam(chain(encoder.parameters(),classifier.parameters()), lr=lr)


    for e in range(10):
        train_model(e, encoder, classifier, train_batch_it,optimizer)

        if e > 1:
            break

    dill.dump(encoder, open("encoder.pkl","wb"))
    dill.dump(classifier, open("classifier.pkl","wb"))


if __name__ == "__main__":
    main()