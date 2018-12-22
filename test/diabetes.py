#!/user/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class DiabetesDataset(Dataset):
    #download, read data
    def __init__(self):
        xy = np.loadtxt('data.csv', delimiter=',', dtype=np.float32)
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
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)


#x_data = Variable(torch.Tensor([[1.0],[2.0],[3.0],[4.0]]))
x_data = Variable(torch.Tensor([[2.1, 0.1],[4.2, 0.8],[3.1, 0.9],[3.3, 0.2]]))
y_data = Variable(torch.Tensor([[0.],[1.],[0.],[1.]]))

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(8, 6)
        self.l2 = torch.nn.Linear(6, 4)
        self.l3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred

model = Model()

#Cross Entropy loss
criterion = torch.nn.BCELoss(size_average=True)#True 返回loss.mean() 否则返回loss.sum()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1) #随机梯度下降

#Training loop
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):#分batch来训练
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        print epoch, i, "inputs", inputs.data, "labels", labels.data
        # forward pass
        y_pred = model(inputs)
        # loss
        loss = criterion(y_pred, labels)
        print epoch, i, loss.data, loss.item()
        # zero gradients, perform a backward pass, updata weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        running_loss += loss.data[0]
        if i%2000 == 1999:
            print '[%d, %5d] loss: %.3f' %(epoch+1, i+1, running_loss / 2000)
            running_loss = 0.0

print 'Finished Training'

