# -*- coding: utf-8 -*-
"""
Created on Wed May  5 16:43:10 2021

@author: holge
"""


if __name__ == "__main__":
    import numpy as np
    import dask.array as da

    A = np.random.random((100, 30, 30))
    B = np.random.random((30, 30))

    Adask = da.from_array(A)
    Bdask = da.from_array(B)

    result = []
    for i in range(100):
        result.append(da.matmul(Adask[i, :, :], Bdask))
    result_out = da.stack(result)
    result_out.visualize(filename='loop.pdf')

    result2_out = da.matmul(Adask, Bdask)
    result2_out.visualize(filename='tensor.pdf')
    
    

