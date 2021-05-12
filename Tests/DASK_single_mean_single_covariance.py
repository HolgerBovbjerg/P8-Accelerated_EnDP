import os

# slow = True
# # slow = False
# if slow:
#     os.environ["MKL_NUM_THREADS"] = "1"

import time
import numpy as np
from dask.distributed import Client, LocalCluster, wait
import dask.array as da


def cov_mult(conv_matrix, cov_matrix):
    conv_matrix = da.transpose(conv_matrix)
    return da.matmul(da.matmul(conv_matrix, cov_matrix), da.transpose(conv_matrix))


def relu(input_samples):
    # da.maximum(input_samples, 0, out=input_samples)
    return da.maximum(input_samples, 0)


if __name__ == '__main__':
    batch_size = 10
    total_kernels = 3
    input_size = 50
    convmatrix = da.random.random((batch_size, 27, input_size ** 2))
    kernels = da.random.random((total_kernels, 3 ** 2 * 3))
    cov_matrices = da.random.random((total_kernels, kernels.shape[1], kernels.shape[1]))
    times = []
    with Client(address='localhost:8001') as client:
        for n in range(1):
            # Do something using 'client'
            convolved_means = []
            convolved_covariances = []
            pre_relu_samples = []
            post_relu_samples = []
            end_means = []
            start = time.time()
            for i in range(batch_size):
                for j in range(total_kernels):
                    convolved_covariances.append(
                        client.submit(
                            cov_mult, convmatrix[i, :, :], cov_matrices[j, :, :]
                        )
                    )
                    convolved_means.append(
                        client.submit(
                            da.matmul, kernels[j, :], convmatrix[i, :, :]
                        )
                    )
            for (mean, covariance) in zip(convolved_means, convolved_covariances):
                pre_relu_samples.append(
                    client.submit(
                        sampling, mean, covariance
                    )
                )
            for samples in pre_relu_samples:
                post_relu_samples.append(
                    client.submit(
                        relu, samples
                    )
                )
            for samples in post_relu_samples:
                end_means.append(
                    da.mean(samples)
                )

            print('submits complete')
            wait(end_means)
            data = client.gather(end_means)
            results = [result.compute for result in data]
            times.append(time.time() - start)

        print('Computation complete! Stopping workers...')

        end = sum(times) / 1
        print(f'Execution completed in {end} seconds')
