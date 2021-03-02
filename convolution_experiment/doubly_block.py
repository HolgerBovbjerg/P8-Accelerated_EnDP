# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 11:13:22 2021

@author: holge
"""


import numpy as np
from scipy.linalg import toeplitz 

# input  signal
I =np.random.rand(10,10)
# filter
F =np.array([[1,0,0], [0, 1, 0], [0,0,1]])

# Dimensions of input and filter
Irownum , Icolnum = I.shape
Frownum , Fcolnum = F.shape

# output diemnsions
outputrownum = Irownum + Frownum - 1
outputcolnum = Icolnum + Fcolnum - 1

# Needed padding
row_padding = outputrownum - Frownum
col_padding = outputcolnum - Fcolnum

# Create zero padded input
F_zero_padded =np.pad(F, ((row_padding , 0),
                        (0, col_padding)),
                        'constant', constant_values=0)

# list for storing individual toeplitz matrices
toeplitz_list = []

# Creating toeplitz matrices
for i in range(F_zero_padded.shape[0]-1,-1,-1):
    c = F_zero_padded[i,:]
    r = np.r_[c[0],np.zeros(Icolnum-1)]
    toeplitz_m = toeplitz(c,r)
    toeplitz_list.append(toeplitz_m)
    print('F'+str(i)+'\n',toeplitz_m)

# Indexes of teoplitz matrices in doubly block
c = range(1,F_zero_padded.shape[0]+1)
r = np.r_[c[0],np.zeros(Irownum-1,dtype=int)]
doubly_ind = toeplitz(c,r)
print('doubly indices\n',doubly_ind)

# Shape of individual toeplitz matrices
toeplitz_shape = toeplitz_list[0].shape

height = toeplitz_shape[0]*doubly_ind.shape[0]
width = toeplitz_shape[1]*doubly_ind.shape[1]

# Shape of doubly block matrix
doubly_blocked_shape = [height,width]
doubly_blocked = np.zeros(doubly_blocked_shape)
b_height, b_width = toeplitz_shape

# Generate doubly block
for i in range(doubly_ind.shape[0]):
    for j in range(doubly_ind.shape[1]):
        start_i = i *b_height
        start_j = j *b_width
        end_i = start_i + b_height
        end_j = start_j + b_width
        doubly_blocked[start_i:end_i,start_j:end_j] = toeplitz_list[doubly_ind[i,j]-1]
        
print('Doubly blocked \n',doubly_blocked)


