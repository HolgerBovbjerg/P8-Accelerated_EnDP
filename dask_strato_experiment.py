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
            results = []
            start = time.time()
            for i in range(batch_size):
                for j in range(total_kernels):
                    results.append(
                        client.submit(
                            cov_mult, convmatrix[i, :, :], cov_matrices[j, :, :]
                        )
                    )
            times.append(time.time() - start)
            print('submits complete')
            wait(results)
            # out = results[0].result()
            # times.append(time.time() - start)
            data = client.gather(results)

        print('Computation complete! Stopping workers...')

        end = sum(times) / 1
        print(f'Execution completed in {end} seconds')


        # Validate that the computed matrices are correct by comparing with numpy locally.
        # np_conv = convmatrix.compute()
        # np_covs = cov_matrices.compute()
        # np_outs = np.empty((total_kernels * batch_size, input_size ** 2, input_size ** 2))
        # for i in range(batch_size):
        #     matrix = np.transpose(np_conv[i, :, :])
        #     for j in range(total_kernels):
        #         np_outs[i * total_kernels + j, :, :] = np.matmul(np.matmul(matrix, np_covs[j, :, :]),
        #                                                          np.transpose(matrix))
        #
        # for i in range(total_kernels * batch_size):
        #     np.allclose(data[i].compute(), np_outs[i])
        #     # print(np.allclose(data[i].compute(), np_outs[i]))
