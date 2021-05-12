import matplotlib.pyplot as plt
import time
from dask.distributed import Client, LocalCluster
import dask.array as da

if __name__ == '__main__':
    cluster = LocalCluster(
    n_workers=2,
    processes=True,
    threads_per_worker=1
    )
    with Client(cluster) as client:
        start = time.time()
        x = da.linspace(0, 10000, 100000)
        y = x * x
        plt.plot(x, y)
        end = time.time() - start
        print('Computation complete! Stopping workers...')

    print(f'Execution completed in {end} seconds')