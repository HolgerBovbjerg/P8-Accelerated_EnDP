# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 11:13:22 2021

@author: holge
"""


import numpy as np
from scipy.linalg import circulant
from scipy import signal

ipython = get_ipython()

def matrixtovector(input):
    input_height , input_width = input.shape
    outputvector = np.zeros(input_height*input_width ,dtype=input.dtype)
    # flip the input matrix up down
    input= np.flipud(input)
    for i,row in enumerate(input):
        start = i*input_width
        end = start + input_width
        outputvector[start:end] = row
    return outputvector

def vectortomatrix(input, output_shape):
    output_height, output_width = output_shape
    print(output_shape)
    output = np.zeros(output_shape, dtype=input.dtype)
    for i in range(output_height):
        start = i*output_width
        end = start + output_width
        output[i, :] =input[start:end]
    # flip the output matrix up−down to get correct result 
    output=np.flipud(output)
    return output
    

# input  signal
# I = np.arange(28**2).reshape(28,28)
I = np.random.rand(28,28)
# filter
F = np.array([[1,0,0], [0, 1, 0], [0,0,1]])

# I = (np.arange(6)+1).reshape(2,3)
# # filter
# F = ((np.arange(4)+1)*10).reshape(2,2)


# Dimensions of input and filter
Irownum , Icolnum = I.shape
Frownum , Fcolnum = F.shape

# output diemnsions
outputrownum = max(Irownum,Icolnum,Fcolnum,Frownum) + (Frownum-1)
outputcolnum = outputrownum 

# Needed padding
Frow_padding = outputrownum - Frownum
Fcol_padding = outputcolnum - Fcolnum

# Create zero padded input
F_zero_padded =np.pad(F, ((Frow_padding , 0),
                        (0, Fcol_padding)),
                        'constant', constant_values=0)

Irow_padding = outputrownum - Irownum
Icol_padding = outputcolnum - Icolnum

# Create zero padded filter
I_zero_padded =np.pad(I, ((Irow_padding , 0),
                        (0, Icol_padding)),
                        'constant', constant_values=0)

print('F_zero_padded\n',F_zero_padded)
print('I_zero_padded\n',I_zero_padded)
# list for storing individual toeplitz matrices
circulant_list = []

# Creating toeplitz matrices
for i in range(F_zero_padded.shape[0]-1,-1,-1):
    c = F_zero_padded[i,:]
    r = np.r_[c[0],np.zeros(Icolnum-1)]
    circulant_m = circulant(c)
    circulant_list.append(circulant_m)
    print('F'+str(2-i)+'\n',circulant_m)

# Indexes of toeplitz matrices in doubly block
c = range(1,F_zero_padded.shape[0]+1)
doubly_ind = circulant(c)
print('doubly indices\n',doubly_ind)

# Shape of individual toeplitz matrices
circulant_shape = circulant_list[0].shape

height = circulant_shape[0]*doubly_ind.shape[0]
width = circulant_shape[1]*doubly_ind.shape[1]

# Shape of doubly block matrix
doubly_blocked_shape = [height,width]
doubly_blocked = np.zeros(doubly_blocked_shape)
b_height, b_width = circulant_shape

# Generate doubly block
for i in range(doubly_ind.shape[0]):
    for j in range(doubly_ind.shape[1]):
        start_i = i *b_height
        start_j = j *b_width
        end_i = start_i + b_height
        end_j = start_j + b_width
        doubly_blocked[start_i:end_i,start_j:end_j] = circulant_list[doubly_ind[i,j]-1]
        
print('Doubly blocked \n',doubly_blocked)

conv_direct =  signal.convolve2d(I,F)

vectorizedI = matrixtovector(I_zero_padded)
print('Vectorized input',vectorizedI)

conv_matmul = np.matmul(doubly_blocked, vectorizedI)
outshape = (outputrownum,outputcolnum)
conv_matmul_matrix = vectortomatrix(conv_matmul,outshape)

# print('Direct convolution\n',conv_direct,'\n')
# print('Mul convolution\n',conv_matmul)
# print('Matrix mul convolution\n',conv_matmul_matrix)
difference = np.max(conv_matmul_matrix - conv_direct)
print('Max difference',difference)

print('Direct convolution')
ipython.magic("timeit signal.convolve2d(I,F)")
print('Matrix product convolution')
ipython.magic("timeit np.matmul(doubly_blocked, vectorizedI)")


