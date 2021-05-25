# -*- coding: utf-8 -*-
"""
Created on Tue May 18 21:40:34 2021

@author: holge

This file is identical to DASK_batch.py except that it creates the input dask-arrays from numpy-arrays
so the random sampling does not show in the task graph when visualizing.
"""


import numpy as np
import dask.array as da

def cov_mult(conv_matrix, cov_matrix):
    conv_matrix = da.transpose(conv_matrix)
    return da.matmul(da.matmul(conv_matrix, cov_matrix), da.transpose(conv_matrix))


def relu(input_samples):
    # da.maximum(input_samples, 0, out= input_samples)  # This might be faster as it puts result in same variable.
    return da.maximum(input_samples, 0)


def random_cov(dim):
    A = np.random.standard_normal(size=(dim, dim)) 
    cov = np.dot(A, A.transpose()) # Ensure pos. semi-definite 
    return cov


def mvn_random_DASK(mean, cov, N, dim):
    da.random.seed(10)
    epsilon = 0.0001
    A = da.linalg.cholesky(cov + epsilon * da.eye(dim), lower=True)
    z = da.random.standard_normal(size=(N, dim))
    x = da.outer(da.ones((N,)), mean).transpose() + da.dot(A, z.transpose())
    return x


if __name__ == '__main__':
    da.random.seed(12)
    batch_size = 2
    input_size = 28
    input_channels = 3
    total_kernels = 5
    total_samples = 1500

    X = np.random.random((batch_size, 3 ** 2 * input_channels, input_size ** 2))
    kernels_mean = np.random.random((total_kernels, 3 ** 2 * input_channels))
    cov_list = [random_cov(3 ** 2 * input_channels) for number in range(total_kernels)]
    kernels_cov = np.stack(cov_list)
    
    X = da.from_array(X)
    kernels_mean = da.from_array(kernels_mean)
    kernels_cov = da.from_array(kernels_cov)

    batch_out = []
    for i in range(batch_size):
        kernel_out = []         
        for j in range(total_kernels):
            mean = da.matmul(kernels_mean[j,:], X[i,:,:])
            cov = da.matmul(da.transpose(X[i,:,:]),
                            da.matmul(kernels_cov[j,:,:], X[i,:,:]))
            z = mvn_random_DASK(mean, cov, total_samples, input_size ** 2)
            g = relu(z)
            mean_g = da.mean(g, axis=1)
            kernel_out.append(mean_g)
        kernels_out = da.stack(kernel_out, axis=0)
        batch_out.append(kernels_out)
    batches_out = da.stack(batch_out, axis=0)  
    print('task graph complete')
    mean_g.visualize(rankdir="LR", 
                     filename="task_graph_mean_g.pdf",
                     cmap='viridis')
    kernels_out.visualize(rankdir="LR", 
                          filename="task_graph_conv_out.pdf")

    batches_out.visualize(rankdir="LR",
                          filename="task_graph_batches_out.pdf")

    