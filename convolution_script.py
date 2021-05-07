# -*- coding: utf-8 -*-
"""
Created on Tue May  4 10:30:52 2021

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
    #
    # a = np.random.random((4000, 4000))
    # b = np.random.random((4000, 4000))
    #
    # start = time.time()
    # for q in range(20):
    #     c = np.matmul(a, b)
    #
    # slut = time.time() - start

    # Sequential
    batch_size = images.shape[0]
    convmatrix = ff.image2convmatrix(torch.tensor(images), kernel_size, padsize)
    mean = ff.create_mean(input_kernels, kernel_size, input_channels)

    print("flag 1")
    start_seq = time.time()
    mu_z = ff.convolution_mean(convmatrix, mean, batch_size, input_kernels)
    out_images = ff.convout2image(mu_z, batch_size, output_channels, output_size)

    total_seq = time.time() - start_seq
    # Parallel
    workers = 8
    print('flag 2')
    start_para = time.time()

    outdaskbatch = ff.DASK_batch_mult(convmatrix, mean, workers, 16, input_size, output_channels)
    total_para = time.time() - start_para
    print('flag 3')
    plt.imshow(out_images[0, 0, :, :], cmap='gray')
    plt.show()

    plt.imshow(outdaskbatch[0, 0, :, :], cmap='gray')
    plt.show()

    print('done')
