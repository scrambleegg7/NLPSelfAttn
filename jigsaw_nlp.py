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

    print("time to build vocab", (time() - start_t))
    print("length of vocaburary", len(TEXT.vocab), TEXT.vocab.vectors.shape )

    print("- "*20 )
    print("* most common words.")
    print( TEXT.vocab.freqs.most_common(20) )

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

    em_sz = 100
    nh = 500
    nl = 3

    model_file = os.path.join(data_dir, "jigsaw_model_7978.pkl")
    model = SimpleBiLSTMBaseline( hidden_dim=nh,emb_dim=em_sz,len_TEXT_vocab=len(TEXT.vocab), v_vec=TEXT.vocab.vectors )

    if os.path.isfile( model_file  ):
        print("model file found.", )
        model.load_state_dict( torch.load( model_file ) )
        #model = dill.load(open(model_file,"rb")) 
        #model = torch
    model.cuda()

    opt = optim.Adam(model.parameters(), lr=1e-4)
    loss_func = nn.BCEWithLogitsLoss()

    epochs = 10
    for epoch in range(1, epochs + 1):
        
        running_loss = 0.0
        #running_corrects = 0
        model.train() # turn on training mode
        
        for idx, (x, y) in enumerate( tqdm.tqdm( train_dl ) ): # thanks to our wrapper, we can intuitively iterate over our data!
            opt.zero_grad()

            preds = model(x)
            loss = loss_func(preds, y)
            loss.backward()
            opt.step()
            
            #if idx % 100 == 0:
            #    print("- "*20)
            #    print("step",idx)
            #    print("preds", preds)
            #    print("loss %.5f" % loss.item())

            running_loss += loss.item() * x.size(0)
            
        epoch_loss = running_loss / len(train)
        
        # calculate the validation loss for this epoch
        val_loss = 0.0
        accs = []
        model.eval() # turn on evaluation mode
        for x, y in valid_dl:
            preds = model(x)
            loss = loss_func(preds, y)
            val_loss += loss.item() * x.size(0)

            logits = preds.cpu().data.numpy()
            logits = 1. / (1. + np.exp( -logits ))
            z = np.zeros_like(logits)
            z[ logits > .5 ] = 1
            y_num = y.cpu().data.numpy()
            acc = (z == y_num).sum() / ( y_num.shape[0] * y_num.shape[1] )
            accs.append( acc )

        val_loss /= len(val)

        print("mean accuracy",   np.mean( accs )  )
        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, epoch_loss, val_loss))

        #dill.dump(model, open("jigsaw_model.pkl","wb"))
        torch.save(model.state_dict(), os.path.join(data_dir, "jigsaw_model_%d.pkl" % epoch ) )



if __name__ == "__main__":
    main()