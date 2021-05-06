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
import Matrix_Vector_mult as mv
from dask.distributed import Client
import dask.array as da
import dask


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
    cov = create_cov(input_kernels, kernel_size, input_channels)
    mu_z = convolution_mean(convmatrix, mean, batch_size, input_kernels)
    Sigma_z = convolution_cov(convmatrix, cov, batch_size, input_kernels)
    return mu_z, Sigma_z


def convolution_layer_distributed(input_data,
                      input_channels,
                      input_size,
                      input_kernels,
                      kernel_size,
                      padsize,
                      output_channels,
                      output_size,
                      client):
    
    batch_size = input_data.shape[0]
    convmatrix = ff.image2convmatrix(torch.tensor(input_data), kernel_size, padsize)
    mean = create_mean(input_kernels, kernel_size, input_channels)
    mu_z = convolution_mean_futures(convmatrix, mean, batch_size, input_kernels, client)
    
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


def convolution_mean_delayed(X, mu_W, batch_size, input_kernels):
    mu_z = np.empty((batch_size, input_kernels, X.shape[2]))
    results = []
    for i in range(batch_size):
        for j in range(input_kernels):
            result = dask.delayed(np.matmul)(mu_W[j, :], X[i, :, :])
            results.append(result)
    mu_z_delayed = dask.persist(*results)
    mu_z_computed = dask.compute(mu_z_delayed)
    for i in range(batch_size):
        for j in range(input_kernels):
            mu_z[i, j, :] = mu_z_computed[0][i*j]
    return mu_z


def convolution_mean_futures(X, mu_W, batch_size, input_kernels, client):
    mu_z = np.empty((batch_size, input_kernels, X.shape[2]))
    for i in range(batch_size):
        for j in range(input_kernels):
            futures = client.submit(np.matmul, mu_W[j, :], X[i, :, :])
    results = client.gather(futures)
    
    return mu_z
  
          
def convolution_cov(X, Sigma_W, batch_size, input_kernels):
    Sigma_z = np.empty((batch_size, input_kernels, X.shape[2], X.shape[2]))
    for i in range(batch_size):
        for j in range(input_kernels):
            Sigma_z[i, j, :, :] = np.matmul(X[i, :, :].transpose(),
                                        np.matmul(Sigma_W[j, :], X[i, :, :])
                                            )
    return Sigma_z


def create_mean(input_kernels, kernel_size, input_channels):
    mu = np.random.rand(input_kernels, kernel_size**2*input_channels)
    return mu


def create_cov(input_kernels, kernel_size, input_channels):
    Sigma = np.random.rand(input_kernels, kernel_size**2*input_channels, kernel_size**2*input_channels)
    return Sigma


def convout2image(convmean, batch_size, output_channels):
    out_images = np.empty((batch_size, output_channels, output_size, output_size))
    for i in range(batch_size):
        for j in range(output_channels):
            out_images[i, j, :, :] = ff.convout2image(mu_z[i, j, :], (output_size, output_size))
    return out_images
    

if __name__ == "__main__":
    #%% Import images
    dir_path = os.path.dirname(os.path.realpath(__file__))
    imagepath = dir_path + "/Resized_images"
    batch_size = 16
    input_size = 32
    input_channels = 3
    
    images = np.empty((batch_size, input_channels, input_size, input_size))
    for i in range(batch_size):
        images[i] = ff.import_image(imagepath, i)
    plt.imshow(images[0, 0, :, :], cmap='gray')
    plt.show()
    #%% Convolution layer 
    padsize = 1
    input_channels = 3
    input_size = 32
    input_kernels = 64
    kernel_size = 3
    output_channels = 64
    output_size = 32
    
    # Sequential
    mu_z, Sigma_z = convolution_layer(images,
                          input_channels,
                          input_size,
                          input_kernels,
                          kernel_size,
                          padsize,
                          output_channels,
                          output_size)
    
    images = convout2image(mu_z, batch_size, output_channels)
    
    plt.imshow(images[0, 0, :, :], cmap='gray')
    plt.show()
    
    # DASK distributed
    # workers = 6
    # client = Client(n_workers=workers)
    
    
    
    # client.close()

    