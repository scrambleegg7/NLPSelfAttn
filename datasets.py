
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

        #self.filenames, self.captions, self.ixtoword, \
        #    self.wordtoix, self.n_words, self.sesitivity = self.load_text_data(data_dir, split)

        vocab_path = os.path.join(data_dir,"imdb.vocab")
        self.ixtoword, self.wordtoix = self.load_vocab(vocab_path)

        self.load_text(split_dir)


        # build dataframe from data list 
        data = {"review" : self.rev_list ,"sensitivity" : self.sensitivity_list }
        self.df = pd.DataFrame.from_dict(data)

        #print(self.df.head())


    def load_text(self,split_dir):

        # define stop words
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

                words_filtered, rev, sensitivity, missing_words_list = self.load_review_files(file_path)

                self.sensitivity_list.append( sensitivity )
                self.rev_list.append( rev )

                if len(missing_words_list) > 0:
                    self.missing_words_total.append( missing_words_list )

            #print("sensitivity", sensitivity)
            #print("review word id", rev)
            #print("words_filtered", words_filtered)
        
        print("missing words number", self.missing_words)
        print("sensitivity", len( self.sensitivity_list ))
        print("review word", len( self.rev_list ))
        
        #print("total missing words." , self.missing_words_total)

    def load_review_files(self, file_path):

        file_name = file_path.split(".")[0]
        sensitivity = file_name.split("_")[-1]

        with open(file_path, "r") as f:
            #captions = f.read().decode('cp437').split('\n')
            review_text = f.read().split('\n')[0]

            reg_token_words = self.tokenizer.tokenize(review_text)
            reg_token_words = [w.lower() for w in reg_token_words  if w.isalpha() ]
            #reg_token_words = [w.lower() for w in reg_token_words ]

            words_filtered = reg_token_words[:] # creating a copy of the words list
            for word in reg_token_words:
                if word in self.stop_words:        
                    words_filtered.remove(word)
                elif word in ["br"]:
                    words_filtered.remove(word)
            
            rev = []
            missing_words_list = []
            for w in words_filtered:
                if w in self.wordtoix:
                    rev.append(self.wordtoix[w])
                else:
                    self.missing_words += 1
                    missing_words_list.append( w )
            
        return words_filtered,rev, sensitivity, missing_words_list


    def load_vocab(self, vocab_path):

        with open(vocab_path, "r") as f:
            #captions = f.read().decode('cp437').split('\n')
            vocab = f.read().split('\n')

            print("total vocabrary", len(vocab))

            ixtoword = {}
            ixtoword[0] = '<end>'
            wordtoix = {}
            wordtoix['<end>'] = 0
            ix = 1
            for w in vocab:
                wordtoix[w] = ix
                ixtoword[ix] = w
                ix += 1


            return ixtoword, wordtoix

    def __getitem__(self, index):
        #
        #print("index from getitem function",index)

        return self.df.iloc[ index, "review"], self.df.iloc[ index, "sensitivity"]


    def __len__(self):

        return len(self.sensitivity_list)





def main():


    data_dir = "/home/donchan/Documents/DATA/IMDB/aclImdb"
    imdb = IMDBTextDataset(data_dir)



if __name__ == "__main__":
    main()