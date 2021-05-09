# -*- coding: utf-8 -*-
"""
Created on Wed May  5 16:43:10 2021

@author: holge
"""

from dask.distributed import Client, wait, progress

import dask.array as da
# import numpy as np
import time


def add(x1, x2):
    return(x1 + x2)


if __name__ == "__main__":
    # client = Client('10.92.0.188:8786')
    client = Client()
    print(client)
    start = time.time()
    # for i in range(100):
    x1 = da.random.random((500, 500), chunks=(100, 100))
    x2 = da.random.random((500, 500), chunks=(100, 100))
    y = da.add(x1, x2)
    
    future = sum(y)
    result = future.compute()
    wait(future)
    stop = time.time()
    client.shutdown()
    # execution_time = stop-start
    # print(execution_time)
