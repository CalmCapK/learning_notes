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
'''
# RNN 1
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

cell = nn.LSTM(input_size=4, hidden_size=2, batch_first=True)
#inputs = torch.autograd.Variable(torch.Tensor([[h, e, l, l, o]]))
#inputs = torch.autograd.Variable(torch.Tensor([[h]]))
inputs = torch.autograd.Variable(torch.Tensor([[h, e, l, l, o],
                                               [e, o, l, l, l],
                                               [l, l, e, e, l]
                                               ]))
print "input size", inputs.size()

hidden = (torch.autograd.Variable(torch.randn(1, 3, 2)), torch.autograd.Variable(
    torch.randn((1, 3, 2))))
out, hidden = cell(inputs, hidden)
print out.data
'''


'''
x = np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
print x[:,0:-1]
print x[:,-1].shape
print x[:,[-1]].shape
print x.shape

#xy = np.loadtxt('data.csv', delimiter=',', dtype=np.float32)
#x_data = Variable(torch.from_numpy(xy[:, 0:-1]))
#y_data = Variable(torch.from_numpy(xy[:, [-1]]))


class DiabetesDataset(Dataset):
    #download, read data
    def __init__(self):
        #xy = np.loadtxt('data.csv', delimiter=',', dtype=np.float32)
        xy = np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
        self.len = xy.shape[0]
        self.x_data = Variable(torch.from_numpy(xy[:, 0:-1]))
        self.y_data = Variable(torch.from_numpy(xy[:, [-1]]))

    #return one item on the index
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    #return the data length
    def __len__(self):
        return self.len

dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=1,
                          shuffle=True,
                          num_workers=2)


for epoch in range(4):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        print epoch, i, "inputs", inputs.data, "labels", labels.data

'''


'''
#model = nn.LogSoftmax()

#model = nn.Softmax()

criterion = nn.NLLLoss()
#input is of size nBatch x mClasses = 3 x 5
inputs = Variable(torch.randn(3,5), requires_grad=True)
#each batch answer 0 <= value <= nclasses
target = Variable(torch.LongTensor([1,0,4]))
#pred = model(inputs)
#model = F.log_softmax()
pred = F.softmax(inputs)
print pred
loss = criterion(pred, target)
l2 = F.nll_loss(pred, target)
print loss, l2
loss.backward()
'''
