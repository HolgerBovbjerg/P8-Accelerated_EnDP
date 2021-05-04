# -*- coding: utf-8 -*-
"""
Created on Tue May  4 10:30:52 2021

@author: holge
"""

import matplotlib.pyplot as plt
import time
import torch
import os
import numpy as np
import function_file as ff

def convolution_layer(input_data,
                      input_channels,
                      input_size,
                      input_kernels,
                      kernel_size,
                      padsize,
                      output_channels,
                      output_size):
    
    batch_size = input_data.shape[0]
    convmatrix = ff.image2convmatrix(torch.tensor(input_data), kernel_size, padsize)
    mean = create_mean(input_kernels, kernel_size, input_channels)
    mu_z = convolution_mean(convmatrix, mean, batch_size, input_kernels)
    out_images = np.empty((batch_size, output_channels, output_size, output_size))
    for i in range(batch_size):
        for j in range(output_channels):
            out_images[i, j, :, :] = ff.convout2image(mu_z[i, j, :], (output_size, output_size))
    
    return out_images

def convolution_mean(X, mu_W, batch_size, input_kernels):
    mu_z = np.empty((batch_size, input_kernels, X.shape[2]))
    for i in range(batch_size):
        for j in range(input_kernels):
            mu_z[i, j, :] = np.matmul(mu_W[j, :], X[i, :, :])
    return mu_z
            
def convolution_cov(X, Sigma_W, batch_size, input_kernels):
    Sigma_z = np.empty((batch_size, input_kernels, X.shape[3], X.shape[3]))
    for i in range(batch_size):
        for j in range(input_kernels):
            Sigma_z[i, j, :, :] = np.matmul(X[i, :, :].transpose(),
                                        np.matmul(Sigma_W[j, :], X[i, :, :])
                                            )

def create_mean(input_kernels, kernel_size, input_channels):
    mu = np.random.rand(input_kernels, kernel_size**2*input_channels)
    return mu

def create_cov(input_kernels, kernel_size, input_channels):
    Sigma = np.random.rand(input_kernels, kernel_size**2*input_channels, kernel_size**2*input_channels)
    return Sigma

if __name__ == "__main__":
    #%% Import images
    dir_path = os.path.dirname(os.path.realpath(__file__))
    imagepath = dir_path + "/Resized_images"
    batch_size = 16
    input_size = 224
    input_channels = 3
    
    images = np.empty((batch_size, input_channels, input_size, input_size))
    for i in range(batch_size):
        images[i] = ff.import_image(imagepath, i)
    plt.imshow(images[1, 1, :, :], cmap='gray')
    plt.show()
    #%% Convolution layer 1
    padsize = 1
    input_channels = 3
    input_size = 224
    input_kernels = 64
    kernel_size = 3
    output_channels = 64
    output_size = 224
    
    out = convolution_layer(images,
                          input_channels,
                          input_size,
                          input_kernels,
                          kernel_size,
                          padsize,
                          output_channels,
                          output_size)
    
    plt.imshow(out[1, 1, :, :], cmap='gray')
    plt.show()