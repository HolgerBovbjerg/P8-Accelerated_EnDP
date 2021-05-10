# -*- coding: utf-8 -*-
"""
Created on Wed May  5 16:43:10 2021

@author: holge
"""

from dask.distributed import Client, wait, progress

import dask.array as da
# import numpy as np
import time


if __name__ == "__main__":
    client = Client('localhost:8001')
    # client = Client()
    print(client)
    print(client.scheduler_info()['services'])
    start = time.time()
    # for i in range(100):
    x1 = da.random.random((50000, 50000), chunks=(1000, 1000))
    
    x2 = da.random.random((50000, 50000), chunks=(1000, 1000))
    y = da.matmul(x1, x2)
    
    future = da.sum(y)
    future.compute()
    wait(future)
    # print(result)
    stop = time.time()
    exec_time= stop - start
    print(f"time: {exec_time:.2f}")
    client.close()

