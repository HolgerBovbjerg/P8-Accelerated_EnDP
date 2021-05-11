# -*- coding: utf-8 -*-
"""
Created on Wed May  5 16:43:10 2021

@author: holge
"""

from dask.distributed import Client, wait, progress

import dask.array as da
import numpy as np
import time
import function_file as ff
import DASK_function_file as dff

if __name__ == "__main__":
    size = 2**5
    N = 1000
    
    print("NumPy:")
    start = time.time()
    mean = np.random.random((size**2, 1))
    A = np.random.random((size**2, size**2))
    cov = np.matmul(A, A.transpose())
    start = time.time()
    y = ff.mvn_random(mean, cov, N)
    stop = time.time()
    exec_time= stop - start
    print(f"time: {exec_time:.2f}")
    
    
    print("DASK:")
    client = Client('localhost:8001')
    # client = Client()
    print(client)
    print(client.scheduler_info()['services'])
    mean = da.random.random((size**2, 1))
    A = da.random.random((size**2, size**2))
    cov = da.matmul(A, A.transpose())
    start = time.time()
    dim = mean.shape[0]
    future = dff.mvn_random_DASK(mean, cov, dim, N)
    y = future.compute()
    stop = time.time()
    exec_time= stop - start
    print(f"time: {exec_time:.2f}")
    client.close()
    
    
    
        # for i in range(100):
    # x1 = da.random.random((50000, 27), chunks=(1000, 1))
    
    # x2 = da.random.random((27, 50000), chunks=(1, 1000))
    # y = da.matmul(x1, x2)
    
    # future = da.sum(y)
    # future.compute()
    # wait(future)
    # print(result)
    
    # for i in range(100):
    # x1 = da.random.random((50000, 27), chunks=(1000, 1))
    
    # x2 = da.random.random((27, 50000), chunks=(1, 1000))
    # y = da.matmul(x1, x2)
    
    # future = da.sum(y)
    # future.compute()
    # wait(future)
    # print(result)
    
    

