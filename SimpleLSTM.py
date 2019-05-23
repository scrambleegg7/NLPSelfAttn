import torch 
import torch.nn as nn 

class SimpleLSTMBaseline(nn.Module):

    def __init__(self, hidden_dim, emb_dim=300, num_linear=1, len_text_vocab=10000):
        super().__init__() # don't forget to call this!
        self.embedding = nn.Embedding(len_text_vocab, emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=1)
        self.linear_layers = []
        for _ in range(num_linear - 1):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.linear_layers = nn.ModuleList(self.linear_layers)
        self.predictor = nn.Linear(hidden_dim, 6)
 
    def forward(self, seq):
        hdn, _ = self.encoder(self.embedding(seq))
        feature = hdn[-1, :, :]
        for layer in self.linear_layers:
          feature = layer(feature)
          preds = self.predictor(feature)
        return preds


class SimpleBiLSTMBaseline(nn.Module):
    def __init__(self, hidden_dim, emb_dim=300,
                 spatial_dropout=0.05, recurrent_dropout=0.1, num_linear=1, len_TEXT_vocab=100000,v_vec=None):
        super().__init__() # don't forget to call this!
        self.embedding = nn.Embedding(len_TEXT_vocab, emb_dim)

        if v_vec is not None:
            self.embedding.weight.data.copy_(v_vec)

        if num_linear == 1:
            self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=1)
        elif num_linear > 2:
            self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=num_linear, dropout=recurrent_dropout)
        self.linear_layers = []
        for _ in range(num_linear - 1):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.predictor = nn.Linear(hidden_dim, 6)
    
    def forward(self, seq):
        hdn, _ = self.encoder(self.embedding(seq))
        feature = hdn[-1, :, :]
        for layer in self.linear_layers:
            feature = layer(feature)
        preds = self.predictor(feature)
        return preds
