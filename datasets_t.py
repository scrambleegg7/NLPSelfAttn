
from glob import glob

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

from torchtext.data import Field, BucketIterator, TabularDataset, ReversibleField
from torchtext.vocab import GloVe
from sklearn.model_selection import train_test_split


class IMDBTextDataset(data.Dataset):
    def __init__(self, data_dir, split='train',
                 base_size=64, emb_num=None):


        self.data = []
        self.data_dir = data_dir
        self.emb_num = emb_num

        # NLTK definition
        self.tokenizer = RegexpTokenizer(r'\w+')     
        self.stop_words = stopwords.words('english')
        self.missing_words = 0

        split_dir = os.path.join(data_dir, split)

        #vocab_path = os.path.join(data_dir,"imdb.vocab")

        text_file = os.path.join( data_dir, "TEXT.pkl" )
        label_file = os.path.join( data_dir, "LABEL.pkl" )

        if os.path.isfile( text_file ):
            TEXT = dill.load( open( text_file, "rb") )
            LABEL = dill.load( open( label_file, "rb") )
        else:
            self.build_dict

        self.load_text(split_dir)

        data_length = len(self.sensitivity_list)
        rand_perm = np.random.permutation(data_length)

        # build dataframe from data list 
        data = {"review" : self.rev_list ,"sensitivity" : self.sensitivity_list }
        self.df = pd.DataFrame.from_dict(data)

        self.df = self.df.iloc[rand_perm].copy()

        train, val = train_test_split(self.df, test_size=0.2)

        train.to_csv( os.path.join(data_dir,"train.csv"), index=False )
        val.to_csv( os.path.join(data_dir,"val.csv"), index=False )

        self.en = spacy.load('en')

        TEXT = Field(sequential=True, tokenize=self.tokenize_en, lower=True)
        LABEL = Field(sequential=False, use_vocab=False)
        data_fields = [('Review', TEXT), ('Sensitivity', LABEL)]

        train, val = TabularDataset.splits(path=data_dir, train='train.csv', 
            validation='val.csv', format='csv', fields=data_fields)

        else:
            TEXT.build_vocab(train, vectors="glove.6B.100d", min_freq=2 )
            LABEL.build_vocab(train)

            # save data field
            dill.dump(TEXT, open( text_file   ,'wb'))
            dill.dump(LABEL, open( label_file  ,'wb'))

        device = torch.device('cuda')
        self.train_iter , self.val_iter = BucketIterator.splits(
                                    (train, val), 
                                batch_size=32, device=device, #sort_key=lambda x:len(x.text),
                                #sort_within_batch=True, 
                                repeat=False)

    def get_train_iter(self):
        return self.train_iter
    
    def get_val_iter(self):
        return self.val_iter

    def next(self):

        batch = next(iter( self.train_iter ))

        return batch

    def tokenize_en(self, sentence):
        soup = BeautifulSoup(sentence, features="lxml")
        clean_text = soup.get_text()
        return [tok.text for tok in self.en.tokenizer(clean_text)]

    def load_text(self,split_dir):

        #sens_dirs = ["neg","pos","unsup"]
        sens_dirs = ["neg","pos"]

        self.sensitivity_list = []
        self.rev_list = []
        self.missing_words_total = []

        for sens in sens_dirs:
            target_dir = os.path.join(split_dir,sens,"*.txt")
            files = glob(target_dir)
            print("total %d files under %s" % (len(files),sens )  )

            for file_path in files:

                review, sensitivity = self.load_review_files(file_path)

                self.sensitivity_list.append( sensitivity )
                self.rev_list.append( review )
        
        print("sensitivity", len( self.sensitivity_list ))
        print("review word", len( self.rev_list ))
        
        #print("total missing words." , self.missing_words_total)

    def load_review_files(self, file_path):

        file_name = file_path.split(".")[0]
        sensitivity = file_name.split("_")[-1]

        with open(file_path, "r") as f:
            #captions = f.read().decode('cp437').split('\n')
            review_text = f.read().split('\n')[0]


            
            
        return review_text, sensitivity








def main():


    data_dir = "/home/donchan/Documents/DATA/IMDB/aclImdb"
    imdb = IMDBTextDataset(data_dir)

    batch = imdb.next()
    print(batch.Review)
    print(batch.Sensitivity)


if __name__ == "__main__":
    main()