# -*- coding: utf-8 -*-
"""
Created on Tue May 11 09:30:03 2021

@author: holge
"""


from dask.distributed import Client, wait, progress

import dask.array as da
import numpy as np
import time
import torch
import function_file as ff



if __name__ == "__main__":
    #%% Setup client
    client = Client('localhost:8001')
    # client = Client()
    
    #%% Input settings
    batch_size = 16
    input_channels = 3
    input_size = 224
    images = np.random.random((batch_size, input_channels, input_size, input_size))
    
    #%% Convolution layer 1
    input_channels = 3
    input_size = 224
    input_kernels = 64
    kernel_size = 3
    output_channels = 64
    output_size = 224
    pad_size = 1 
    
    convmatrix = ff.image2convmatrix(torch.tensor(images), kernel_size, pad_size)
    
    W1 = np.random.rand(input_kernels, kernel_size, kernel_size)
    W1_vec = np.empty((input_kernels, kernel_size**2*input_channels))
    for i in range(input_kernels):
        W1_vec[i, :] = ff.filt2vec(W1[i, :, :], input_channels)
    
    
    
    
    out_dask = []
    dask_convmatrix = []
    W1_dask = []
    for i in range(batch_size):
        for j in range(input_kernels):
            dask_convmatrix.append(da.from_array(convmatrix[i, :, :], chunks=(input_size**2, 1)))
            W1_dask.append(da.from_array(W1_vec[j, :]))
            client.submit(da.matmul(W1_dask[input_kernels*i + j], dask_convmatrix))
   
    results = client.gather(out_dask)
    
    # out1 = np.empty((batch_size, output_channels, input_size**2))
    # for i in range(batch_size):
    #     for j in range(input_kernels):
    #         print(j*(i+1))
    #         out1[i, j, :, :] = out_dask[(i+1)*j]

    
    client.close()
    