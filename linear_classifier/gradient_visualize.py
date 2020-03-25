#!/usr/bin/python3

import matplotlib.pyplot as plt
from random import random as randn
import numpy as np

from multiclass_svm_loss import L, L_vec

# Generate dummy input vector
# x= np.random.rand(3072)
x= np.array([-15,22,-44,56])
x= np.append(x,1)

# correct class
y= 2

# Generate dummy weighting matrix W
# W= np.random.rand(10, 3072)
b= np.array([0.0,0.2,-0.3])
W= np.array([[0.01,-0.05,0.1,0.05],[0.7,0.2,0.05,0.16],[0,-0.45,-0.2,0.03]])
W= np.append(W, np.array([b]).T, axis=1)

# print(L(np.array([x,x]).T, np.array([y,y]), W))

# Generate random directions to move
W_1= np.random.rand(*W.shape)
print(W_1)

l_seq= np.zeros(10)
a_seq= np.arange(l_seq.shape[0])*0.02

for idx in range(l_seq.shape[0]):
    # Compute SVM loss
    a= a_seq[idx]
    W_now= (W+ a*W_1)
    loss= L_vec(x, y, W_now)
    print(loss)
    l_seq[idx]= loss


plt.plot(a_seq, l_seq)
plt.show()