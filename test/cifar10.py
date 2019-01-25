#!/user/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import torch as t
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
        self.x_data = Variable(t.from_numpy(xy[:, 0:-1]))
        self.y_data = Variable(t.from_numpy(xy[:, [-1]]))

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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5) #转为1位
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = t.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        print epoch, i, "inputs", inputs.data, "labels", labels.data

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        if i%2000 == 1999:
            print '[%d, %5d] loss: %.3f' %(epoch+1, i+1, running_loss / 2000)
            running_loss = 0.0

print 'Finished Training'
