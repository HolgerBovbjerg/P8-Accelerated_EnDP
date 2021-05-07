# -*- coding: utf-8 -*-
"""
Created on Wed May  5 16:43:10 2021

@author: holge
"""

from dask.distributed import Client, wait

import dask.array as da
import numpy as np
import time

def add(x1, x2):
    return(x1 + x2)

if __name__ == "__main__":
    
    # client = Client('10.92.0.188:8786') 
    client = Client
    print(client)
    # start = time.time()
    # for i in range(100):
    x1 = da.random.random((50000, 50000), chunks=(1000, 1000))
    x2 = da.random.random((50000, 50000), chunks=(1000, 1000))
    future = client.submit(da.add, x1, x2)

    # client.close()
    # # # wait(y)
    # # stop = time.time()
    # execution_time = stop-start
    # print(execution_time)
