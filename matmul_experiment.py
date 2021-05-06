# -*- coding: utf-8 -*-
"""
Created on Wed May  5 09:11:54 2021

@author: holge
"""

import numpy as np
import time
from numba import njit, prange


@njit(parallel=True)
def naiveMatMul(A, B):
    rowsA = A.shape[0]
    # colsA = A.shape[1]
    rowsB = B.shape[0]
    colsB = B.shape[1]

    C = np.zeros((rowsA, colsB))

    for i in prange(rowsA):
        for j in prange(colsB):
            for k in prange(rowsB):
                C[i, j] += A[i, k]*B[k, j]
    return C


def blockedMatMul(A, B, blocksize):
    rowsA = A.shape[0]
    # colsA = A.shape[1]
    rowsB = B.shape[0]
    colsB = B.shape[1]
    C = np.zeros((rowsA, colsB))
    T = blocksize
    for i in range(int(rowsA/T)):
        for j in range(int(colsB/T)):
            for k in range(int(rowsB/T)):
                C[i*T:i*T+T, j*T:j*T+T] += np.matmul(A[i*T:i*T+T, k*T:k*T+T],
                                                     B[k*T:k*T+T, j*T:j*T+T])
    return C


def gotoMatMulGEMM(A, B, panelsize, blocksize):
    rowsA = A.shape[0]
    colsA = A.shape[1]
    rowsB = B.shape[0]
    colsB = B.shape[1]
    C = np.zeros((rowsA, colsB))
    P = panelsize
    for i in range(int(colsA/P)):
        for j in range(int(rowsB/P)):
            C += gotoMatMulGEPP(A[:, P*i:P*(i+1)],
                                B[:, P*j:P*(j+1)],
                                blocksize)
    return C


def gotoMatMulGEPP(Apanel, Bpanel, blocksize):
    rowsA = Apanel.shape[0]
    colsA = Apanel.shape[1]
    rowsB = Bpanel.shape[0]
    colsB = Bpanel.shape[1]
    B = blocksize
    C = np.zeros((rowsA, colsB))
    for i in range(int(rowsA/B)):
        C += gotoMatmulGEBP(A[B*i:B*(i+1)], Bpanel)
    return C
        
def gotoMatmulGEBP(Ablock, Bpanel):
    rowsA = Ablock.shape[0]
    colsA = Ablock.shape[1]
    rowsB = Bpanel.shape[0]
    colsB = Bpanel.shape[1]
    for i in range 
        



if __name__ == "__main__":
    rowsA = 2**10
    colsA = 2**10
    rowsB = colsA
    colsB = 2**10

    A = np.random.rand(rowsA, colsA)
    B = np.random.rand(rowsB, colsB)

    mc = 1
    nr = 1
    kc = 1

    start = time.time()
    Cnp = np.matmul(A, B)
    time_numpy = time.time() - start

    # start = time.time()
    # Cnaive = naiveMatMul(A,B)
    # time_naive = time.time() - start

    start = time.time()
    blocksize = 64
    Cblocked = blockedMatMul(A, B, blocksize)
    time_blocked = time.time() - start

    print(np.allclose(Cnp, Cblocked))
