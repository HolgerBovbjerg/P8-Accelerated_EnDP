# -*- coding: utf-8 -*-
"""
Created on Tue May 11 09:56:18 2021

@author: holge
"""

import dask.array as da



def mvn_random_DASK(mean, cov, dim, N):
    A = da.linalg.cholesky(cov)
    N = 1000
    z = da.random.normal(0, 1, (dim, N))   
    x = da.add(mean,da.dot(A,z))
    return x

