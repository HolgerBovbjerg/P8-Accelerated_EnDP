# -*- coding: utf-8 -*-
"""
Created on Mon May  3 10:39:49 2021

@author: holge
"""
import os

import numpy as np
import torch
from torchvision import datasets, transforms
from dask.distributed import Client
import dask.array as da
import dask

def import_image(imagefolder, imagenumber):
    """
    imports image from imagefolder

    Parameters
    ----------
    imagefolder : str
        path to images.
    imagenumber : int
        image number.

    Returns
    -------
    image : Tensor
        image from imagefolder as PyTorch Tensor

    """
    transform = transforms.ToTensor()
    dataset = datasets.ImageFolder(imagefolder, transform=transform)
    image = dataset[imagenumber][0]
    image = image[None, :, :, :]
    return image


def image2convmatrix(image, kernelsize, padsize):
    """
    Converts image Tensor into convolution matrix and converts to numpy array

    Parameters
    ----------
    image : Tensor
        Imput image
    kernelsize : int
        size of convolution kernel
    padsize : int
        Amount of zero padding

    Returns
    -------
    ndarray
        Convolution matrix

    """
    unfold = torch.nn.Unfold(kernel_size=kernelsize, padding=padsize)
    convmatrix = unfold(image)
    return np.array(convmatrix)


def convout2image(matrix, imagedims):
    """
    converts matmul convolution result to image

    Parameters
    ----------
    matrix : ndarray
        matmul convolution result
    imagedims : tuple
        tuple containing original image resolution

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return matrix.reshape(imagedims)


def filt2vec(W, dims):
    """
    Turns filter kernel matrix into vector

    Parameters
    ----------
    W : ndarray
        Filter kernel weight matrix
    dims : int
        number of input channels (e.g. 3 for RGB images)

    Returns
    -------
    W_vec : ndarray
        Vectorized convolution kernel

    """
    size = W.shape
    # Convert the filter into a convolution kernel adapted to the convolution operation
    W_temp = W.reshape((1, 1, size[0], size[1]))
    # Convolution output channel
    W_temp = np.repeat(W, dims, axis=1)
    # Make into tensor and "reshape" into one vector
    W_vec = np.reshape(W_temp, (1, (size[0] ** 2) * dims))
    return W_vec


def convolution_mean(X, mu_W, batch_size, input_kernels):
    mu_z = np.empty((batch_size, input_kernels, X.shape[2]))
    for i in range(batch_size):
        for j in range(input_kernels):
            mu_z[i, j, :] = np.matmul(mu_W[j, :], X[i, :, :])
    return mu_z

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
    convmatrix = image2convmatrix(torch.tensor(input_data), kernel_size, padsize)
    mean = create_mean(input_kernels, kernel_size, input_channels)
    mu_z = convolution_mean_futures(convmatrix, mean, batch_size, input_kernels, client)
    
    out_images = np.empty((batch_size, output_channels, output_size, output_size))
    for i in range(batch_size):
        for j in range(output_channels):
            out_images[i, j, :, :] = convout2image(mu_z[i, j, :], (output_size, output_size))

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
    mu = np.random.rand(input_kernels, kernel_size ** 2 * input_channels)
    return mu


def create_cov(input_kernels, kernel_size, input_channels):
    Sigma = np.random.rand(input_kernels, kernel_size ** 2 * input_channels, kernel_size ** 2 * input_channels)
    return Sigma


def convout2image(convmean, batch_size, output_channels, output_size):
    out_images = np.empty((batch_size, output_channels, output_size, output_size))
    for i in range(batch_size):
        for j in range(output_channels):
            out_images[i, j, :, :] = convout2image(convmean[i, j, :], (output_size, output_size))
    return out_images

def DASK_batch_mult(matrix_input, vector_input, workers, batch_size, input_size, output_channels):
    client = Client(n_workers=workers)
    results = []
    batch_no = matrix_input.shape[0] // batch_size

    for i in range(batch_no):
        batch = client.scatter(matrix_input[i * batch_size: i * batch_size + batch_size])
        results.append(
            client.submit(convolution_mean, batch, vector_input, batch_size, vector_input.shape[0])
        )

    client.gather(results)
    out_tensor = np.zeros((batch_size * batch_no, output_channels, input_size, input_size))
    for i in range(batch_no):
        out_tensor[i * batch_size: i * batch_size + batch_size] = results[i].result().reshape(batch_size,
                                                                                              output_channels,
                                                                                              input_size, input_size)

    return out_tensor