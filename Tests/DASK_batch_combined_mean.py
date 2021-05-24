# -*- coding: utf-8 -*-
"""
Created on Tue May 18 21:40:34 2021

@author: holge
"""


import time
import numpy as np
from dask.distributed import Client, LocalCluster, wait, config
import dask.array as da
import dask

import validation_functions as va


def cov_mult(conv_matrix, cov_matrix):
    conv_matrix = da.transpose(conv_matrix)
    return da.matmul(da.matmul(conv_matrix, cov_matrix), da.transpose(conv_matrix))


def relu(input_samples):
    # da.maximum(input_samples, 0, out= input_samples)  # This might be faster as it puts result in same variable.
    return da.maximum(input_samples, 0)


def random_cov_DASK(dim):
    A = da.random.standard_normal(size=(dim, dim))
    cov = da.dot(A, A.transpose())
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
    # Simulation settings
    itrs = 5
    batch_size = 5
    input_size = 28
    input_channels = 256
    total_kernels = 512
    total_samples = 1500
    
    # Generate random input
    X = da.random.random((batch_size, 3 ** 2 * input_channels, input_size ** 2))
    kernels_mean = da.random.random((total_kernels, 3 ** 2 * input_channels))
    cov_list = [random_cov_DASK(3 ** 2 * input_channels) for number in range(total_kernels)]
    kernels_cov = da.stack(cov_list)
    
    # Validation function call
    # validate =True
    validate = False
    if validate:
        numpy_validation_list = va.single_mean_single_covariance_validator(X.compute(), kernels_mean.compute(),
                                                                           kernels_cov.compute(), batch_size, total_kernels,
                                                                           input_size)
    
    times = [] # list for storing execution times
    cluster = 'localhost:8001' # address of compute cluster
    with Client(cluster) as client: # Using cluster as client do
        for n in range(itrs): # itrs runs 
            start = time.time() # save start tikme
            batch_out = [] # create list for batch output
            for i in range(batch_size): # for each image
                kernel_out = [] # create list for kernel outputs
                mean = da.matmul(kernels_mean, X[i,:,:]) # compute all kernel means
                for j in range(total_kernels): # for each kernel
                    cov = da.matmul(da.transpose(X[i,:,:]), # compute covariance
                                    da.matmul(kernels_cov[j,:,:], X[i,:,:]))
                    z = mvn_random_DASK(mean[j,:], cov, total_samples, input_size ** 2) # sample from transformed distribution
                    g = relu(z) # pass samples through relu
                    mean_g = da.mean(g, axis=1) # compute ensemble mean from samples
                    kernel_out.append(mean_g) # add ensemble mean to kernel outputs list 
                kernels_out = da.stack(kernel_out, axis=0) # stack all kernel outputs
                batch_out.append(kernels_out) # add stacked kernel outputs to batch output list
            batches_out = da.stack(batch_out, axis=0) # stack output from each image 
            print('task graph complete')
            batches_out_result = batches_out.compute() # compute task graph
            print("compute done")
            times.append(time.time() - start) # save execution time to list
            if validate: # if validate then test correctness of result
                print(
                    f'are the results within 0.02 of eachother? {np.allclose(numpy_validation_list, atol=0.02)}')

        print('Computation complete! Stopping workers...')

        times_mean = sum(times) / 5
        print(f'Execution completed in {times_mean} seconds')

