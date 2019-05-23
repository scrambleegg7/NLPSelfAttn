import torch
import torch.nn as nn 
import numpy as np 
import os 
import sys 

import torch.nn.functional as F
from torch.autograd import Variable

class Attn(nn.Module):
    def __init__(self, h_dim):
        super(Attn, self).__init__()
        self.h_dim = h_dim
        self.main = nn.Sequential(
            nn.Linear(h_dim, 24),
            nn.ReLU(True),
            nn.Linear(24,1)
        )

    def forward(self, encoder_outputs):
        b_size = encoder_outputs.size(0)
        attn_ene = self.main(encoder_outputs.contiguous().view(-1, self.h_dim)) # (b, s, h) -> (b * s, 1)


        return F.softmax(attn_ene.contiguous().view(b_size, -1), dim=1).unsqueeze(2) # (b*s, 1) -> (b, s, 1)


class Attn2(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn2, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            #self.attn = nn.Linear(self.hidden_size, hidden_size)

            d_a = 100
            r = 10

            self.linear_first = torch.nn.Linear(hidden_size,d_a)
            self.linear_first.bias.data.fill_(0)
            self.linear_second = torch.nn.Linear(d_a,r)
            self.linear_second.bias.data.fill_(0)
            #self.n_classes = n_classes


            #self.linear_final = torch.nn.Linear(lstm_hid_dim,self.n_classes)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, encoder_outputs):
    
        x = torch.tanh(self.linear_first( encoder_outputs ))
        x = self.linear_second(x)
        #x = self.softmax(x,1)    
    
        return x 
    
    def score(self, hidden, encoder_output):
        
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy
        
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.dot(energy)
            return energy




class AttnClassifier(nn.Module):
    def __init__(self, h_dim, c_num):
        super(AttnClassifier, self).__init__()
        self.attn = Attn(h_dim)
        self.main = nn.Linear(h_dim, c_num)
        
    
    def forward(self, encoder_outputs):
        attns = self.attn(encoder_outputs) #(b, s, 1)

        #print("attns shape"  , attns.shape )
        feats = (encoder_outputs * attns).sum(dim=1) # (b, s, h) -> (b, h)
        #print("feats shape", feats.shape)
        #return F.log_softmax(self.main(feats)), attns
        feats_ = self.main(feats)
        return nn.LogSoftmax(dim=1)(feats_), attns