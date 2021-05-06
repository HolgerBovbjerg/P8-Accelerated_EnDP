# -*- coding: utf-8 -*-
"""
Created on Mon May  3 10:39:49 2021

@author: holge
"""
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import torch
from torchvision import datasets, transforms


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
