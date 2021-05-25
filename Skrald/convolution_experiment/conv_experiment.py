# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 10:25:34 2021

@author: holge
"""


import numpy as np

from timeit import default_timer as timer
from scipy import signal,linalg


X = np.array([[2,5,3],[1,4,1]])
print("X:\n",X)
W= np.array([[1,2],[3,4]])
print("W:\n",W)
t_start = timer()
out1 = signal.convolve2d(X,W)
t_end = timer()
duration_conv = (t_end - t_start)*1e6
print("X*W:\n",out1)

Xvector = np.array([2,5,3,1,4,1])
print("X vectorized:\n",Xvector)

result_size0 = W.shape[0] + X.shape[0] - 1
result_size1 = W.shape[1] + X.shape[1] - 1

padsize0 = result_size0 -  W.shape[0]
padsize1 = result_size1 -  W.shape[1]

padding0 = np.zeros(padsize0, W.dtype)
padding1 = np.zeros(padsize1, W.dtype)

H = np.pad(W,((0,padsize0),(0,padsize1)), 'constant')

first_row = np.r_[H[0,0], padding1]
first_col = H[0,:]
H0 = linalg.toeplitz(first_col, first_row)

first_row = np.r_[H[1,0], padding1]
first_col = H[1,:]
H1 = linalg.toeplitz(first_col, first_row)

first_row = np.r_[H[2,0], padding1]
first_col = H[2,:]
H2 = linalg.toeplitz(first_col, first_row)

Hlist = [H0, H1, H2]

Wblocktoeplitz = np.concatenate((np.concatenate((Hlist[0],Hlist[2]),axis=1),
                                np.concatenate((Hlist[1],Hlist[0]),axis=1),
                                np.concatenate((Hlist[2],Hlist[1]),axis=1)),axis=0)



Wtoeplitz = np.array([[1,0,0,0,0,0],
                      [2,1,0,0,0,0],
                      [0,2,1,0,0,0],
                      [0,0,2,0,0,0],
                      [3,0,0,1,0,0],
                      [4,3,0,2,1,0],
                      [0,4,3,0,2,1],
                      [0,0,4,0,0,2],
                      [0,0,0,3,0,0],
                      [0,0,0,4,3,0],
                      [0,0,0,0,4,3],
                      [0,0,0,0,0,4]
                      ]) 
print("W toeplitz:\n",Wtoeplitz)

t_start = timer()
out2 = np.dot(Wtoeplitz,Xvector)
t_end = timer()
duration_dot = (t_end - t_start)*1e6

Wvector = np.array([1,2,3,4])
# Xtoeplitz = 