# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 22:32:48 2021

@author: holge
"""

import numpy as np
from scipy.linalg import circulant

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
    
def to_circulant_doubly_block(X, W):
    # Dimensions of input and weights
    Xrownum , Xcolnum = X.shape
    Wrownum , Wcolnum = W.shape
    
    # output diemnsions
    outputrownum = max(Xrownum,Xcolnum,Wcolnum,Wrownum) + (Wrownum-1)
    outputcolnum = outputrownum 
    
    # Needed padding
    Wrow_padding = outputrownum - Wrownum
    Wcol_padding = outputcolnum - Wcolnum
    
    # Create zero padded input
    W_zero_padded =np.pad(W, ((Wrow_padding , 0),
                        (0, Wcol_padding)),
                        'constant', constant_values=0)
    Xrow_padding = outputrownum - Xrownum
    Xcol_padding = outputcolnum - Xcolnum

    # Create zero padded filter
    X_zero_padded =np.pad(X, ((Xrow_padding , 0),
                            (0, Xcol_padding)),
                            'constant', constant_values=0)
    
    circulant_list = []
    # Creating circulant matrices
    for i in range(W_zero_padded.shape[0]-1,-1,-1):
        c = W_zero_padded[i,:]
        circulant_m = circulant(c)
        circulant_list.append(circulant_m)
    
    # Indexes of circulant matrices in doubly block
    c = range(1,W_zero_padded.shape[0]+1)
    doubly_ind = circulant(c)
    
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
     
    return [doubly_blocked, matrixtovector(X_zero_padded)]
