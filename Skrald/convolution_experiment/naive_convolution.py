# -*- coding: utf-8 -*-
"""
Created on Mon May  3 09:20:48 2021

@author: holge
"""

import numpy as np
import torch


def Naive_Mult(A, B):
    """
    Naive matrix multiplication

    Parameters
    ----------
    A : ndarray
        input Matrix
    B : ndarray
        input Matrix

    Returns
    -------
    ndarray
        Matrix product of A an B

    """
    # dimension check
    if A.shape[1] == B.shape[0]:
        # allocate memory for result
        result = np.zeros((A.shape[0], B.shape[1]))
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(B)):
                    result[i, j] += A[i, k] * B[k, j]

        return result

    else:
        return print('Dimensions must match')


if __name__ == "__main__":
    X = torch.rand(224, 224)
    W = torch.rand(224, 5)
    Z = Naive_Mult(X, W)
    Z2 = np.matmul(X, W)

    print(np.allclose(Z, Z2))
