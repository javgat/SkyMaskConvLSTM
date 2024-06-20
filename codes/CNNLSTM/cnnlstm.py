import itertools
import random

import torch
from torch import nn

from vae import VAE

class CNNLSTM(nn.Module):
    def __init__(self, cnnvae: VAE, z_dim, hidden_dim, num_layers=1, device='cpu'):
        super(CNNLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cnnvae = cnnvae
        self.device = device
        # LSTM as Seq2Seq LSTM model to customize output length.
        self.lstm_enc = nn.LSTM(z_dim, hidden_dim, num_layers, batch_first=True).to(device)
        self.lstm_dec = nn.LSTM(z_dim, hidden_dim, num_layers, batch_first=True).to(device)
        self.fc = nn.Linear(hidden_dim, z_dim).to(device)


    def forward(self, video, target_len: int,  target_seq=None, teacher_forcing_ratio=0.5):
        # b, t, c, i0, i1
        video = video.to(self.device)
        if target_seq is not None:
            target_seq = target_seq.to(self.device)
        seq_dims = video.shape[0:2]
        video = video.flatten(0,1)
        zvects, _, _ = self.cnnvae.encode(video)
        zvects = torch.unflatten(zvects, 0, seq_dims)
        ## LSTM
        _, hidden_lstm_s2s = self.lstm_enc(zvects)
        batch_size = zvects.size(0)
        target_size = zvects.size(2)
        lstm_dec_input = zvects[:, -1].unsqueeze(1)
        lstm_out = torch.zeros((batch_size, target_len, target_size)).to(zvects.device)
        for t in range(target_len):
            out, hidden_lstm_s2s = self.lstm_dec(lstm_dec_input, hidden_lstm_s2s)
            out = self.fc(out)
            lstm_out[:, t, :] = out.squeeze(1)
            if target_seq is not None and random.random() < teacher_forcing_ratio:
                lstm_dec_input, _, _ = self.cnnvae.encode(target_seq[:, t, :])
                lstm_dec_input = lstm_dec_input.unsqueeze(1)
            else:
                lstm_dec_input = out
        #lstm_out = self.fc(lstm_out)
        lstm_out = lstm_out.flatten(0,1)
        ##
        predicted = self.cnnvae.decode(lstm_out)
        predicted = torch.unflatten(predicted, 0, (seq_dims[0], target_len))
        return predicted

    def parameters(self, recurse: bool = True):
        if recurse:
            return itertools.chain(self.lstm_enc.parameters(), self.lstm_dec.parameters(), self.fc.parameters())
        return []
