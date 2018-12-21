#!/user/bin/python
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[1.0],[2.0],[3.0]]))
y_data = Variable(torch.Tensor([[2.0],[4.0],[6.0]]))

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = Model()
criterion = torch.nn.MSELoss(size_average=True)#True 返回loss.mean() 否则返回loss.sum()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) #随机梯度下降

#Training loop
for epoch in range(500):
    #forward pass
    y_pred = model(x_data)
    #loss
    loss = criterion(y_pred, y_data)
    print epoch, loss.data, loss.item()
    #zero gradients, perform a backward pass, updata weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

hour_val = Variable(torch.Tensor([[4.0]]))
print 4, model.forward(hour_val).data[0][0]