# -*- coding: utf-8 -*-
"""
Created on Sun May  9 21:06:07 2021

@author: holge
"""

import os

slow = True
# slow = False
if slow:
    os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import time
import torch
import numpy as np
from dask.distributed import Client
import function_file as ff

if __name__ == "__main__":
    # %% Import images
    dir_path = os.path.dirname(os.path.realpath(__file__))
    imagepath = dir_path + "/Resized_images"
    batch_size = 128
    input_size = 224
    input_channels = 3

    images = np.random.random((batch_size, input_channels, input_size, input_size))
    # images = np.empty((batch_size, input_channels, input_size, input_size))
    # for i in range(batch_size):
    #     images[i] = ff.import_image(imagepath, i)
    # plt.imshow(images[0, 0, :, :], cmap='gray')
    # plt.show()
    # %% Convolution layer
    padsize = 1
    input_channels = 3
    input_size = 224
    input_kernels = 64
    kernel_size = 3
    output_channels = 64
    output_size = 224

    # test
    
    a = np.random.random((112**2, 112**2))
    b = np.random.random((112**2, 112**2))
    
    start = time.time()
    # for q in range(20):
    c = np.matmul(a, b)
    
    slut = time.time() - start

    # Sequential
    batch_size = images.shape[0]
    convmatrix = ff.image2convmatrix(torch.tensor(images), kernel_size, padsize)
    mu_W = ff.create_mean(input_kernels, kernel_size, input_channels)

    print("Sequential")
    start_seq = time.time()
    mu_z = ff.convolution_mean(convmatrix, mu_W, batch_size, input_kernels)
    # out_images = ff.convout2image(mu_z, batch_size, output_channels, output_size)
    total_seq = time.time() - start_seq


    # print("Parallel")
    # client = Client(n_workers = 12)
    # start_par= time.time()
    # mu_z = ff.convolution_mean_futures(convmatrix, mu_W, batch_size, input_kernels, client)
    # # out_images = ff.convout2image(mu_z, batch_size, output_channels, output_size)
    # total_par = time.time() - start_par

    
    print('done')