# -*- coding: utf-8 -*-
"""
Created on Tue May 18 16:31:40 2021

@author: holge
"""

import dask.array as da

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

batch_size = 5
input_size = 28
input_channels = 3
total_kernels = 5
N = 1000
X = da.random.random((batch_size, 
                      3 ** 2 * input_channels, 
                      input_size ** 2))
kernels_mean = da.random.random((total_kernels, 
                                 3 ** 2 * input_channels))
kernels_cov = da.random.random((total_kernels, 
                                 3 ** 2 * input_channels, 
                                 3 ** 2 * input_channels))

result = []
for i in range(batch_size):
    results = []         
    for j in range(total_kernels):
        mean = da.matmul(kernels_mean[j,:], X[i,:,:])
        cov = da.matmul(da.transpose(X[i,:,:]),
                        da.matmul(kernels_cov[j,:,:], X[i,:,:]))
        z = mvn_random_DASK(mean, cov, N, input_size ** 2)
        g = relu(z)
        results.append(da.mean(g, axis=1))
    result.append(da.stack(results, axis=0))
out = da.stack(result, axis=0)   



# result2 = da.matmul(X,kernels_mean)
