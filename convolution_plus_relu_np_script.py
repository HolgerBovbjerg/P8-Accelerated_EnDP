# -*- coding: utf-8 -*-
"""
Created on Tue May 11 20:00:06 2021

@author: holge
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import time
import torch

import function_file as ff
import Matrix_Vector_mult as mv


if __name__ == "__main__":
    #%% Import images
    dir_path = os.path.dirname(os.path.realpath(__file__))
    imagepath = dir_path + "/Resized_images"
    batch_size = 16
    input_size = 32
    input_channels = 3
    # X = np.empty((batch_size, 27, 224**2))
    # for i in range(batch_size):
    #     image = ff.import_image(imagepath, i)
    #     X[i, :, :] = ff.image2convmatrix(image, 3, 1)
    
    
    #%% torch functions
    relu = torch.nn.ReLU()
    
    #%% Convolution layer 1
    # Settings
    input_channels = 3
    input_size = 32
    input_kernels = 64
    kernel_size = 3
    output_channels = 64
    output_size = 32
    samples = 10
    
    X = np.random.random((batch_size, kernel_size**2*input_channels, input_size**2))
    
    # Create weight vector
    mean_W1 = np.random.random((input_kernels, kernel_size**2*input_channels))
    A = np.random.random((input_kernels, kernel_size**2*input_channels, kernel_size**2*input_channels))
    cov_W1 = np.empty((input_kernels, kernel_size**2*input_channels, kernel_size**2*input_channels))
    for j in range(input_kernels):
        cov_W1[j, :, :] = np.cov(A[j, :, :])
    
    start = time.time()
    
    mean_z1 = np.empty((batch_size, output_channels, input_size**2))
    cov_z1 = np.empty((batch_size, output_channels, input_size**2, input_size**2))
    z1 = np.empty((batch_size, output_channels, input_size**2, samples))
    g1 = np.empty((batch_size, output_channels, input_size**2, samples))
    mean_g1 = np.empty((batch_size, output_channels, input_size**2))
    cov_g1 = np.empty((batch_size, output_channels, input_size**2, input_size**2))
    out_images1 = np.empty((batch_size, output_channels, output_size, output_size))
    
    # Convolution
    for i in range(batch_size):
        for j in range(input_kernels):
            mean_z1[i, j, :] = np.matmul(mean_W1[j, :], X[i, :, :])
            cov_z1[i, j, :, :] = np.matmul(X[i, :, :].transpose(), np.matmul(cov_W1[j, :, :], X[i, :, :]))
            z1[i, j, :, :] = np.random.multivariate_normal(mean_z1[i, j, :], cov_z1[i, j, :, :], samples).transpose()
    
    # ReLU
    g1 = np.array(relu(torch.tensor(z1))) 
    
    # Mean and cov from ReLU
    for i in range(batch_size):
            for j in range(input_kernels):        
                mean_g1[i, j, :] = np.mean((g1[i, j, :, :]))
                cov_g1[i, j, :, :] = np.cov((g1[i, j, :, :]))
    
    # Output image
    out_images1 = ff.convout2image(mean_z1, batch_size, output_channels, output_size)
    execution_time_conv1 = (time.time() - start)
    
    
    plt.imshow(out_images1[1, 1, :, :], cmap='gray')
    plt.show()
    #%%    
    # test_image = ff.convout2image(g1[:, :, :, 1], batch_size, output_channels, output_size)
    # plt.imshow(test_image[1, 1, :, :], cmap='gray')
    
        #%% ReLU
    # relu = torch.nn.ReLU()
    # input size = 32
    # sample_size = 1000
    # z = np.empty((batch_size, output_channels, input_size**2, sample_size))
    # for i in range(batch_size):
    #     for j in range(output_channels):
    #         z[i, j, :, :] = np.random.multivariate_normal(mu_conv1,
    #                                                       Sigma_conv1,
    #                                                       1000)
    # g = np.array(relu(torch.tensor(z)))
    # mu_g = mean(g)
    # Sigma_g = cov(g) 
    # ReLU_out1 = 