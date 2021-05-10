# -*- coding: utf-8 -*-
"""
Created on Mon May 10 09:10:58 2021

@author: holge
"""

import os

slow = True
# slow = False
if slow:
    os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import time
import torch
import numpy as np
from dask.distributed import Client
import function_file as ff

if __name__ == "__main__":

    # test
    sizeA = 124**2
    mean = np.random.normal(0, 1, (sizeA, 1))
    Q = np.random.normal(0, 1, (sizeA, sizeA))
    Qtran = Q.transpose()
    # eig_mean = 2
    # D = np.diag(np.abs(eig_mean + np.random.normal(0, 1, (sizeA, 1)) )[:,0])
    # A = np.matmul(Q,D,D)
    cov = np.matmul(Q,Qtran)
    # b = np.random.random((112**2, 112**2))
    
    start = time.time()
    
    A = np.linalg.cholesky(cov)
    N = 1000
    z = np.random.normal(0, 1, (sizeA, N))   
    x = np.add(mean,np.dot(A,z))
    
    slut = time.time() - start
