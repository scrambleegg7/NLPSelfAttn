
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


class IMDBCSV(data.Dataset):
    def __init__(self, data_dir, split='train'):


        self.data_dir = data_dir

        split_dir = os.path.join(data_dir, split)
        self.load_text(split_dir)

        data = {"review" : self.rev_list ,"sensitivity" : self.sensitivity_list }
        self.df = pd.DataFrame.from_dict(data)
        self.df.to_csv( os.path.join(data_dir,"imdb_data_train.csv"), index=False )


        split_dir = os.path.join(data_dir, "test")
        self.load_text(split_dir)

        data = {"review" : self.rev_list ,"sensitivity" : self.sensitivity_list }
        self.df = pd.DataFrame.from_dict(data)
        self.df.to_csv( os.path.join(data_dir,"imdb_data_test.csv"), index=False )


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
    imdb = IMDBCSV(data_dir)



if __name__ == "__main__":
    main()