# -*- coding: utf-8 -*-
"""
Created on Tue May 18 21:40:34 2021

@author: holge
"""

import os

# slow = True
# # slow = False
# if slow:
#     os.environ["MKL_NUM_THREADS"] = "1"

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
    batch_size = 5
    input_size = 28
    input_channels = 256
    total_kernels = 512
    total_samples = 1500

    X = da.random.random((batch_size, 3 ** 2 * input_channels, input_size ** 2))
    kernels_mean = da.random.random((total_kernels, 3 ** 2 * input_channels))
    cov_list = [random_cov_DASK(3 ** 2 * input_channels) for number in range(total_kernels)]
    kernels_cov = da.stack(cov_list)

    # validate =True
    validate = False
    if validate:
        numpy_validation_list = va.single_mean_single_covariance_validator(X.compute(), kernels_mean.compute(),
                                                                           kernels_cov.compute(), batch_size, total_kernels,
                                                                           input_size)

    times = []
    with Client('localhost:8001') as client:
        for n in range(5):
            # Do something using 'client'
            start = time.time()
            conv_out = []
            for i in range(batch_size):
                kernel_out = []         
                for j in range(total_kernels):
                    mean = da.matmul(kernels_mean[j,:], X[i,:,:])
                    cov = da.matmul(da.transpose(X[i,:,:]),
                                    da.matmul(kernels_cov[j,:,:], X[i,:,:]))
                    z = mvn_random_DASK(mean, cov, total_samples, input_size ** 2)
                    g = relu(z)
                    kernel_out.append(da.mean(g, axis=1))
                conv_out.append(da.stack(kernel_out, axis=0))
            batch_out = da.stack(conv_out, axis=0)  
            print('task graph complete')
            # batch_out.visualize()
            out = batch_out.compute()
            print("compute done")
            times.append(time.time() - start)
            if validate:
                print(
                    f'are the results within 0.02 of eachother? {np.allclose(numpy_validation_list, atol=0.02)}')

        print('Computation complete! Stopping workers...')

        end = sum(times) / 5
        print(f'Execution completed in {end} seconds')
