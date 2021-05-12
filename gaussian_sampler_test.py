# -*- coding: utf-8 -*-
"""
Created on Mon May 10 09:10:58 2021

@author: holge
"""



import time
import numpy as np
import function_file as ff

if __name__ == "__main__":

    # # test
    # sizeA = 124**2
    # mean = np.random.normal(0, 1, (sizeA, 1))
    # Q = np.random.normal(0, 1, (sizeA, sizeA))
    # Qtran = Q.transpose()
    # # eig_mean = 2
    # # D = np.diag(np.abs(eig_mean + np.random.normal(0, 1, (sizeA, 1)) )[:,0])
    # # A = np.matmul(Q,D,D)
    # cov = np.matmul(Q,Qtran)
    # # b = np.random.random((112**2, 112**2))
    
    # start = time.time()
    
    # A = np.linalg.cholesky(cov)
    # N = 1000
    # z = np.random.normal(0, 1, (sizeA, N))   
    # x = np.add(mean,np.dot(A,z))
    
    # slut = time.time() - start
    
    cov = ff.random_cov(10)
    eigenvalues = np.linalg.eigvals(cov)
    x = ff.mvn_random(np.zeros((10,1)), cov, 100)
