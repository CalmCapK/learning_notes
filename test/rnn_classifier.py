#!/user/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


N_CHARS = 128
HIDDEN_SIZE = 100
N_CLASSES = 18

#GPU
def create_variable(tensor):
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)



class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 n_layers=1):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        print self.embedding
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        batch_size = input.size(0)
        input = input.t()
        print " input", input.size()
        embeded = self.embedding(input)
        print " embedding", embeded.size()
        hidden = self._init_hidden(batch_size)
        output, hidden = self.gru(embeded, hidden)
        print " gru hidden output", hidden.size()
        fc_output = self.fc(hidden)
        return fc_output

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        return Variable(hidden)
        #return create_variable(hidden) #GPU

def str2ascii_arr(name):
    arr = [ord(c) for c in name]
    return arr, len(arr)

def pad_sequences(vectorized_seqs, seq_lengths):
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
    #return create_variable(seq_tensor) #GPU
    return seq_tensor

def make_variables(names):
    sequence_and_length = [str2ascii_arr(name) for name in names]
    vectorized_seqs = [sl[0] for sl in sequence_and_length]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length])
    return pad_sequences(vectorized_seqs, seq_lengths)




if __name__ == '__main__':
    names = ['adylov', 'solan', 'hard', 'san']
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_CLASSES)
    print "Let's use", torch.cuda.device_count(), "GPUs!"
    if torch.cuda.device_count() > 1:
        print "Let's use", torch.cuda.device_count(), "GPUs!"
        classifier = nn.DataParallel(classifier)
    #GPU
    #if torch.cuda.is_available():
    #   classifier.cuda()



    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

   # for name in names:
    #    arr, _ = str2ascii_arr(name)
     #   inp = Variable(torch.LongTensor([arr]))
      #  out = classifier(inp)
       # print "in", inp.size(), "out", out.size()

    inputs = make_variables(names)
    print inputs
    out = classifier(inputs)
    print out
    print "batch in", inputs.size(), 'batch out', out.size()
