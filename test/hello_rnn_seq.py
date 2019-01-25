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



idx2char = ['h', 'i', 'e', 'l', 'o']
x_data = [0, 1, 0, 2, 3, 3]
y_data = [1, 0, 2, 3, 3, 4]
one_hot_lookup = [[1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]

inputs = torch.autograd.Variable(torch.Tensor([x_one_hot]))
labels = torch.autograd.Variable(torch.LongTensor(y_data))

num_classes = 5
input_size = 5 # one-hot size
hidden_size = 5
batch_size = 1 #one sentenxe
sequence_length = 6
num_layers = 1

class RNN(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          batch_first=True)

    def forward(self, x):
        hidden = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        x = x.view(x.size(0), self.sequence_length, self.input_size)
        out, _ = self.rnn(x, hidden)
        out = out.view(-1, num_classes)
        return out

    def init_hidden(self):
        return torch.autograd.Variable(torch.zeros(num_layers, batch_size, hidden_size))




loss = 0
rnn = RNN(num_classes, input_size, hidden_size,
                   num_layers)

#print rnn
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.1)

for epoch in range(100):
    outputs = rnn(inputs)
    optimizer.zero_grad()
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    _, idx = outputs.max(1)
    idx = idx.data.numpy()
    result_str = [idx2char[c] for c in idx.squeeze()]
    print "epoch: %d, loss: %1.3f" % (epoch + 1, loss.item())
    print "Predicted string: ", ''.join(result_str)

