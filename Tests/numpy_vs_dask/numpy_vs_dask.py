# -*- coding: utf-8 -*-
"""
Created on Sun May 16 11:24:26 2021

@author: holge

Test script which tests the performance of numpy versus dask
The following operationms are tested:
    dot product
    matrix multiplication
    cholesky decomposition
"""

import numpy as np
import dask.array as da
import time

import pandas as pd
import seaborn as sns


from threadpoolctl import threadpool_limits

from dask.distributed import Client, LocalCluster


    #%% start local dask cluster
if __name__ == '__main__':
    cluster = LocalCluster(n_workers = 4, threads_per_worker = 3)
    # cluster = 'localhost:8001'
    client = Client(cluster)
    # client = Client()


    #%% Dot test
    # with threadpool_limits(limits=1, user_api='blas'):
    x = 10000
    y = 10000
    itr = 1
    A = np.random.random((x,y))
    B = np.random.random((y,1))
    
    start = time.time()
    for i in range(itr):
        C = np.dot(A,B)    
    execution_time_np_dot = (time.time() - start)/itr
    
    Adask = da.from_array(A, chunks = 'auto')
    Bdask = da.from_array(B, chunks = 'auto')
    start = time.time()
    for i in range(itr):    
        C = da.dot(Adask,Bdask)    
        result = C.compute()
    execution_time_dask_dot = (time.time() - start)/itr
    
    #%% Matmul test
    # with threadpool_limits(limits=1, user_api='blas'):
        
    x = 10000
    y = 10000
    z = 10000
    
    itr = 1
    
    A = np.random.random((x,y))
    B = np.random.random((y,z))
    
    start = time.time()
    for i in range(itr):
        C = np.matmul(A,B)    
    execution_time_np_matmul = (time.time() - start)/itr
     
    Adask = da.from_array(A, chunks = 'auto')
    Bdask = da.from_array(B, chunks = 'auto')
    start = time.time()
    for i in range(itr): 
        C = da.matmul(Adask,Bdask) 
        result = C.compute()
    execution_time_dask_matmul = (time.time() - start)/itr
        
    #%% Cholesky test
    # with threadpool_limits(limits=1, user_api='blas'):
    #     x = 10000
        
    #     itr = 1
        
    #     A = np.random.random((x,x))
    #     A = np.matmul(A,A.transpose())
    #     start = time.time()
    #     for i in range(itr):
    #         B = np.linalg.cholesky(A)    
    #     execution_time_np_cholesky = (time.time() - start)/itr
        
    #     Adask = da.from_array(A, chunks = 'auto')
    #     start = time.time()
    #     for i in range(itr):
    #         B = da.linalg.cholesky(Adask)    
    #         result  = B.compute()
    #     execution_time_dask_cholesky = (time.time() - start)/itr
    
    # client.close()
    
    #%% plots
    execution_times = [execution_time_np_dot, 
                        execution_time_np_matmul, 
                        #execution_time_np_cholesky,
                        execution_time_dask_dot,
                        execution_time_dask_matmul] #,
                        #execution_time_dask_cholesky]
    
    data_set = {'Operation': ["dot", "matmul", # "cholesky",
                           "dot", "matmul"], # , "cholesky"],
            'Library': ["np", "np", #"np", 
                        "dask", "dask"],#, "dask"],
            'Execution time': execution_times}
    
    df = pd.DataFrame(data_set)
    
    sns.color_palette("deep")
    
    g = sns.catplot(x="Operation", y='Execution time', hue="Library", kind="bar", data=df)
    # g.set(yscale="log")
    g.set(ylabel = "Execution time [s]")