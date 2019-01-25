#!/user/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
x=np.array([2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1])
y=np.array([2.4,0.7,2.9,2.2,3,2.7,1.6,1.1,1.6,0.9])
mean_x=np.mean(x)
mean_y=np.mean(y)
scaled_x=x-mean_x
scaled_y=y-mean_y
data=np.matrix([[scaled_x[i],scaled_y[i]] for i in range(len(scaled_x))])
plt.plot(scaled_x, scaled_y, 'o')
plt.ylabel('y')
plt.xlabel('x')
plt.show()
cov = np.cov(scaled_x, scaled_y)
eig_val, eig_vec = np.linalg.eig(cov)
print eig_val
print eig_vec
print eig_vec[:,1]
#降维
eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
print eig_pairs
eig_pairs.sort(reverse=True)
print eig_pairs
feature=eig_pairs[0][1]
new_data_reduced=np.transpose(np.dot(feature,np.transpose(data)))
print new_data_reduced
