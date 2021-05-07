# -*- coding: utf-8 -*-
"""
Created on Tue May  4 10:30:52 2021

@author: holge
"""
import os

import matplotlib.pyplot as plt
import time
import torch
import os
import numpy as np
import function_file as ff
import Matrix_Vector_mult as mv
from dask.distributed import Client
import dask.array as da
import dask


if __name__ == "__main__":
    # %% Import images
    dir_path = os.path.dirname(os.path.realpath(__file__))
    imagepath = dir_path + "/Resized_images"
    batch_size = 100
    input_size = 32
    input_channels = 3

    images = np.empty((batch_size, input_channels, input_size, input_size))
    for i in range(batch_size):
        images[i] = ff.import_image(imagepath, i)
    plt.imshow(images[0, 0, :, :], cmap='gray')
    plt.show()
    # %% Convolution layer
    padsize = 1
    input_channels = 3
    input_size = 32
    input_kernels = 64
    kernel_size = 3
    output_channels = 64
    output_size = 32

    # # Sequential
    # mu_z, Sigma_z = convolution_layer(images,
    #                       input_channels,
    #                       input_size,
    #                       input_kernels,
    #                       kernel_size,
    #                       padsize,
    #                       output_channels,
    #                       output_size)
    
    # images = convout2image(mu_z, batch_size, output_channels)
    
    # plt.imshow(images[0, 0, :, :], cmap='gray')
    # plt.show()
    
    # DASK distributed
    # workers = 6
    # client = Client(n_workers=workers)
    
    
    
    # client.close()

    
    # out = convolution_layer(images,
    #                         input_channels,
    #                         input_size,
    #                         input_kernels,
    #                         kernel_size,
    #                         padsize,
    #                         output_channels,
    #                         output_size)

    # Sequential
    batch_size = images.shape[0]
    convmatrix = ff.image2convmatrix(torch.tensor(images), kernel_size, padsize)
    mean = ff.create_mean(input_kernels, kernel_size, input_channels)

    print("flag 1")
    start_seq = time.time()
    mu_z = ff.convolution_mean(convmatrix, mean, batch_size, input_kernels)
    out_images = np.empty((batch_size, output_channels, output_size, output_size))
    for i in range(batch_size):
        for j in range(output_channels):
            out_images[i, j, :, :] = ff.convout2image(mu_z[i, j, :], (output_size, output_size))
    total_seq = time.time() - start_seq
    # Parallel
    workers = 8
    print('flag 2')
    start_para = time.time()
    # outdaskblock = mv.DASK_block_mult(convmatrix,
    #                                   mean,
    #                                   workers,
    #                                   input_size,
    #                                   kernel_size,
    #                                   input_channels,
    #                                   batch_size,
    #                                   output_channels)

    outdaskbatch = mv.DASK_batch_mult(convmatrix, mean, workers, 10, input_size, output_channels)
    total_para = time.time() - start_para
    print('flag 3')
    plt.imshow(out_images[0, 0, :, :], cmap='gray')
    plt.show()

    plt.imshow(outdaskbatch[0, 0, :, :], cmap='gray')
    plt.show()

    print('done')
