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
    batch_size = 3
    input_size = 28
    input_channels = 256
    total_kernels = 512
    total_samples = 1500

    convmatrix = da.random.random((batch_size, 3 ** 2 * input_channels, input_size ** 2))
    kernels = da.random.random((total_kernels, 3 ** 2 * input_channels))
    cov_list = [random_cov_DASK(kernels.shape[1]) for number in range(total_kernels)]
    cov = da.stack(cov_list)

    # validate =True
    validate = False
    if validate:
        numpy_validation_list = va.single_mean_single_covariance_validator(convmatrix.compute(), kernels.compute(),
                                                                           cov.compute(), batch_size, total_kernels,
                                                                           input_size)

    times = []
    with Client(address='localhost:8001') as client:
        for n in range(5):
            # Do something using 'client'
            start = time.time()
            out_image_list = []
            convolved_means = {}
            convolved_covariances = {}
            pre_relu_samples = {}
            post_relu_samples = {}
            end_means = {}
            for i in range(batch_size):
                convolved_means[f'{i}'] = []
                convolved_covariances[f'{i}'] = []
                pre_relu_samples[f'{i}'] = []
                post_relu_samples[f'{i}'] = []
                end_means[f'{i}'] = []

                for j in range(total_kernels):
                    convolved_covariances[f'{i}'].append(
                        cov_mult(convmatrix[i, :, :], cov[j, :, :])
                    )
                    convolved_means[f'{i}'].append(
                        da.matmul(kernels[j, :], convmatrix[i, :, :])
                    )
                for (mean, covariance) in zip(convolved_means[f'{i}'], convolved_covariances[f'{i}']):
                    pre_relu_samples[f'{i}'].append(
                        mvn_random_DASK(mean, covariance, total_samples, input_size ** 2)
                    )
                for samples in pre_relu_samples[f'{i}']:
                    post_relu_samples[f'{i}'].append(
                        relu(samples)
                    )
                for samples in post_relu_samples[f'{i}']:
                    end_means[f'{i}'].append(
                        samples.mean(axis=1)
                    )

                out_image_list.append(
                    end_means[f'{i}']
                )


            out_combined = da.stack(out_image_list, axis=0)
            print('submits complete')
            results = out_combined.compute()
            times.append(time.time() - start)

            if validate:
                print(
                    f'are the results within 0.02 of eachother? {np.allclose(numpy_validation_list, out_combined, atol=0.02)}')

        print('Computation complete! Stopping workers...')

        end = sum(times) / 5
        print(f'Execution completed in {end} seconds')
