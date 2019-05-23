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
from jigsaw_data import dataset
from EncRNN import EncoderRNN2, EncoderRNN
from attn import Attn, Attn2

from attn_models import Attention, AttentionClassifier

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

def train_model(epochs, encoder, classifier, optimizer ,train_dl, legth_of_train):

    encoder.train()
    classifier.train()
    correct = 0


    loss_func = nn.BCEWithLogitsLoss()

    for e in epochs:

        running_loss = 0.0
        for idx, (x, y) in enumerate( tqdm.tqdm( train_dl ) ): # thanks to our wrapper, we can intuitively iterate over our data!
            

            outputs, hidden = encoder(x)
            if encoder.bidirectional: # need to concat the last 2 hidden layers
                hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
            else:
                hidden = hidden[-1]

            logits, attn = classifier(outputs, hidden)


            optimizer.zero_grad()


            loss = loss_func(logits, y)
            loss.backward()
            optimizer.step()
            
            #if idx % 100 == 0:
            #    print("- "*20)
            #    print("step",idx)
            #    print("preds", preds)
            #    print("loss %.5f" % loss.item())

            running_loss += loss.item() * x.size(0)
            #print(running_loss)   
        epoch_loss = running_loss / legth_of_train
        print("epoch %d loss %.5f" % (e, epoch_loss))




def main():


    train, val, TEXT, LABEL = dataset()

    legth_of_train = len(train)

    train_iter, val_iter = BucketIterator.splits(
        (train, val), # we pass in the datasets we want the iterator to draw data from
        batch_sizes=(batch_size,batch_size),
        device=torch.device("cuda"), # if you want to use the GPU, specify the GPU number here
        sort_key=lambda x: len(x.comment_text), # the BucketIterator needs to be told what function it should use to group the data.
        sort_within_batch=False,
        repeat=False # we pass repeat=False because we want to wrap this Iterator layer.
    )

    train_dl = BatchWrapper(train_iter, "comment_text", ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
    valid_dl = BatchWrapper(val_iter, "comment_text", ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])

    x,y = next( iter( train_dl ) )
    print("test sentence shape", x.shape)
    #print(y.shape)

    nh = TEXT.vocab.vectors.shape[1]
    emb_dim = 100
    h_dim = 100

    enc2 = EncoderRNN2(input_size=len(TEXT.vocab), hidden_size=nh, v_vec=TEXT.vocab.vectors, n_layers=2)
    attn = Attention(200,200,200)
    classifier = AttentionClassifier(200,6)


    if torch.cuda.is_available():
        enc2.cuda()
        attn.cuda()
        classifier.cuda()

    #
    #
    lr = 0.001
    optimizer = optim.Adam(chain(enc2.parameters(),classifier.parameters()), lr=lr)
    #
    output, hidden = enc2(x)
    print(output.shape, hidden.shape)

    if enc2.bidirectional: # need to concat the last 2 hidden layers
        print("encorder bidirectional is selected.")
        print("hidden[-1]", hidden[-1])
        print("hidden[-2]", hidden[-2])
        print("the above 2 hidden are concatenated.")
        hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
    else:
        hidden = hidden[-1]

    print("final hidden shape",hidden.shape)

    epochs = range(10)
    train_model(epochs, enc2, classifier, optimizer, train_dl, legth_of_train)

if __name__ == "__main__":
    main()