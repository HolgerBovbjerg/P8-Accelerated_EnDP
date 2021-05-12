import os

slow = True
# slow = False
if slow:
    os.environ["MKL_NUM_THREADS"] = "1"

import time
from dask.distributed import Client, LocalCluster, wait
import multiprocessing as mp
import dask.array as da

import function_file as ff

if __name__ == '__main__':
    batch_size = 100
    total_kernels = 80
    convmatrix = da.random.random((batch_size, 27, 224 ** 2))
    kernels = da.random.random((total_kernels, 3 ** 2 * 3))
    times = []
    with LocalCluster(
            n_workers=1,
            # n_workers=int(0.9 * mp.cpu_count()),
            processes=True,
            threads_per_worker=1,
            # memory_limit='2GB',
            ip='tcp://localhost:9895',
    ) as cluster, Client(cluster) as client:
        for n in range(10):
            # Do something using 'client'
            start = time.time()
            results = []
            for i in range(batch_size):
                for j in range(total_kernels):
                    results.append(
                        client.submit(
                            da.matmul, kernels[j, :], convmatrix[i, :, :]
                        )
                    )
            wait(results)
            # times.append(time.time() - start)
            data = client.gather(results)
            times.append(time.time() - start)

        print('Computation complete! Stopping workers...')

    end = sum(times) / 10
    print(f'Execution completed in {end} seconds')
