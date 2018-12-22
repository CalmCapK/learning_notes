#!/user/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]



'''
import math
import matplotlib.pyplot as plt
z_data = list()
m_data = list()

for i in np.arange(-20.0, 20.0, 0.01):
    z = 1.0/(1.0+math.e**(-i))
    z_data.append(z)
    print i,z
    m_data.append(i)
plt.plot(m_data, z_data)
plt.ylabel('z')
plt.xlabel('x')
plt.show()
'''


w = Variable(torch.Tensor([1.0]), requires_grad=True)
'''
x = Variable(torch.ones(2), requires_grad=True)
z = 4*x*x
y = z.norm()
y.backward()
print x.grad
print x.data
'''
#x = Variable(torch.randn(2,10)) #生成2*10维数组
#print x
print w

def forward(x):
    return x*w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred-y)*(y_pred-y)

def gradient(x, y):
    return 2*x*(x*w-y)

print "predict (before training)", 4, forward(4)

for epoch in range(10):
    for x,y in zip(x_data, y_data):
        #old
        #grad = gradient(x, y)
        #w1 = w.data - 0.01 * grad.data
        #print grad.data,w1.data
        l = loss(x, y)
        l.backward()
        w.data = w.data - 0.01 * w.grad.data
        print "\t", x, y, w.grad.data[0], w.data[0]
        w.grad.data.zero_() #必须归零

    print "process:", epoch, l.data

print "predict (after training)", 4, forward(4)

