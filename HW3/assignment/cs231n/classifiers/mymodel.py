import numpy as np
import torch
import torch.nn as nn

# Modeled over rnn.py's CaptioningRNN


class MyModel(nn.Module):
    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128, hidden_dim=128):
        super(MyModel, self).__init__()

        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)

        # Initialize layers
        self.embedding = nn.Embedding(vocab_size, wordvec_dim)
        self.linear_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(wordvec_dim, hidden_dim, batch_first=True)
        self.linear_vocab = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, features, captions):
        input_captions = self.embedding(captions) #  N * T * D
        h_init = self.linear_proj(features) # N * H
        h_init.unsqueeze_(0) # 1 * N * H
        c_init = torch.zeros_like(h_init) # 1 * N * H
        output, _ = self.lstm(input_captions, (h_init, c_init)) 
        output = self.linear_vocab(output)
        return output