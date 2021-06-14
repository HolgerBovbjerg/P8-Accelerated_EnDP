# -*- coding: utf-8 -*-
"""
Created on Mon May 24 15:05:44 2021

@author: holge
"""
import numpy as np
import dask.array as da
from dask.distributed import Client
import time

import pandas as pd
import seaborn as sns
if __name__ == '__main__':
    # cluster = LocalCluster(n_workers = 4, threads_per_worker = 3)
    cluster = 'localhost:8001'
    client = Client(cluster)
            
    itr1 = 4
    itr2 = 3
    execution_time_dask_dot = []
    execution_time_dask_matmul = []
    for i in range(itr1):
        x = 1000*i + 1000
        y = 1000*i + 1000
        z = 1000*i + 1000

        A = np.random.random((x,y))
        B = np.random.random((y,z))
        
        Adask = da.from_array(A, chunks = 'auto')
        Bdask = da.from_array(B, chunks = 'auto')
    
        start = time.time()
        results = []
        for j in range(itr2): 
            for k in range(z):
                results.append(da.matmul(Adask, Bdask[:,k]))
            results_stacked = da.stack(results)    
            result = results_stacked.compute()
        execution_time_dask_dot.append((time.time() - start)/itr2)
        
        
        start = time.time()
        for j in range(itr2): 
            C = da.matmul(Adask,Bdask) 
            result2 = C.compute()
        execution_time_dask_matmul.append((time.time() - start)/itr2)
    
    #%%
    data_set = {'Operation': ["matrix-vector", "matrix-vector", "matrix-vector", "matrix-vector", # "cholesky",
                           "matrix-matrix", "matrix-matrix", "matrix-matrix", "matrix-matrix"], # , "cholesky"],
            'n': [1000, 2000, 3000, 4000,
                  1000, 2000, 3000, 4000],
            'Execution time [s]':  execution_time_dask_dot + execution_time_dask_matmul}
    
    df = pd.DataFrame(data_set)
    
    sns.color_palette("deep")
    
    fig = sns.relplot(x="n", y='Execution time [s]', hue="Operation", kind="line", data=df)
    fig.savefig("DASK_matmul_test.pdf", bbox_inches='tight')
    