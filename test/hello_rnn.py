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


#hello rnn complete

idx2char = ['h', 'i', 'e', 'l', 'o']
x_data = [0, 1, 0, 2, 3, 3]
y_data = [1, 0, 2, 3, 3, 4]
one_hot_lookup = [[1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]

inputs = torch.autograd.Variable(torch.Tensor(x_one_hot))
labels = torch.autograd.Variable(torch.LongTensor(y_data))

num_classes = 5
input_size = 5 # one-hot size
hidden_size = 5
batch_size = 1 #one sentenxe
sequence_length = 1
num_layers = 1

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          batch_first=True)

    def forward(self, hidden, x):
        x = x.view(batch_size, sequence_length, input_size)
        #hidden:(batch, num_layers*num_directions, hidden_size)
        out, hidden = self.rnn(x, hidden)
        out = out.view(-1, num_classes)
        return hidden, out

    def init_hidden(self):
        return torch.autograd.Variable(torch.zeros(batch_size, num_layers, hidden_size))

loss = 0
model = Model()
#crossEntropyLoss = logSoftmax + NULLLoss
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(100):
    optimizer.zero_grad()
    loss = 0
    hidden = model.init_hidden()
    sys.stdout.write("predicted string: ")
    for input, label in zip(inputs, labels):
        hidden, output = model(hidden, input)
        val, idx = output.max(1)
        sys.stdout.write(idx2char[idx.data[0]])
        label = torch.tensor([label.data])

        loss += criterion(output, label)

    print ", epoch: %d, loss: %1.3f" %(epoch + 1, loss.item())
    loss.backward()
    optimizer.step()