import torch
import torch.nn as nn 
import numpy as np 
import os 
import sys 


from torch.autograd import Variable


class EncoderRNN2(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1, v_vec=None, bidirectional=True):
        super(EncoderRNN2, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        if v_vec is not None:
            self.embedding.weight.data.copy_(v_vec)

        if n_layers > 1: # layers should be 2.
            self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=bidirectional)
        else:
            self.gru = nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=bidirectional)

    def forward(self, input_seqs, input_lengths=None, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        packed = embedded

        # for the moment we do not use this function
        #if input_lengths is not None:
        #    packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        
        outputs, hidden = self.gru(packed, hidden)

        #print("gru hidden shape to be sum for bidirectional", hidden.shape)

        #if input_lengths is not None:
        #    outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack (back to padded)
        
        # concatenate should be used for LSTM to offset double size.
        #outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs
        return outputs, hidden

class EncoderRNN(nn.Module):

    def __init__(self, emb_dim, h_dim, v_size, gpu=True, v_vec=None, batch_first=True):
        super(EncoderRNN, self).__init__()
        self.gpu = gpu
        self.h_dim = h_dim
        self.embed = nn.Embedding(v_size, emb_dim)
        if v_vec is not None:
            self.embed.weight.data.copy_(v_vec)

        self.lstm = nn.LSTM(emb_dim, h_dim, batch_first=batch_first,
                            bidirectional=True)

    def init_hidden(self, b_size):
        h0 = Variable(torch.zeros(1*2, b_size, self.h_dim))
        c0 = Variable(torch.zeros(1*2, b_size, self.h_dim))
        if self.gpu:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return (h0, c0)

    def forward(self, sentence, lengths=None):
        self.hidden = self.init_hidden(sentence.size(0))
        emb = self.embed(sentence)
        packed_emb = emb

        #print("packed_emb shape", packed_emb.shape)

        if lengths is not None:
            lengths = lengths.view(-1).tolist()
            packed_emb = nn.utils.rnn.pack_padded_sequence(emb, lengths)

        out, hidden = self.lstm(packed_emb, self.hidden)
        #print("out shape", out.shape)
        #print("hidden shape", hidden.shape)


        if lengths is not None:
            out = nn.utils.rnn.pad_packed_sequence(out)[0]

        out = out[:, :, :self.h_dim] + out[:, :, self.h_dim:]
        #print("final out shape", out.shape)

        return out



def main():

    emb_dim = 200
    h_dim = 300
    v_size = 40000

    rrnn = EncoderRNN2(v_size, h_dim)

if __name__ == "__main__":


    main()