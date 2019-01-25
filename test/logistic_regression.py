#!/user/bin/python
# -*- coding: utf-8 -*-

# complete

import torch
from torch.autograd import Variable
import numpy as np

#x_data = Variable(torch.Tensor([[1.0],[2.0],[3.0],[4.0]]))
x_data = Variable(torch.Tensor([[2.1, 0.1],[4.2, 0.8],[3.1, 0.9],[3.3, 0.2]]))
y_data = Variable(torch.Tensor([[0.],[1.],[0.],[1.]]))

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out1 = self.sigmoid(self.linear(x))
        y_pred = self.sigmoid(self.linear2(out1))
        return y_pred

model = Model()

#Cross Entropy loss
criterion = torch.nn.BCELoss(size_average=True)#True 返回loss.mean() 否则返回loss.sum()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) #随机梯度下降

#Training loop
for epoch in range(100):
    #forward pass
    y_pred = model(x_data)
    #loss
    loss = criterion(y_pred, y_data)
    print epoch, loss.data, loss.item()
    #zero gradients, perform a backward pass, updata weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#for i in np.arange(0.0, 10.0, 0.1):
   # hour_val = Variable(torch.Tensor([[i]]))
   # print hour_val.data, model.forward(hour_val).data[0][0]
#hour_val = Variable(torch.Tensor([[0.5]]))
hour_val = Variable(torch.Tensor([[0.5,1]]))
print hour_val.data, model.forward(hour_val).data[0][0]
#hour_val = Variable(torch.Tensor([[7.0]]))
hour_val = Variable(torch.Tensor([[4.,0.1]]))
print hour_val.data, model.forward(hour_val).data[0][0]