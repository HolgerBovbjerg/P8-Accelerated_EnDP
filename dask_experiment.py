# -*- coding: utf-8 -*-
"""
Created on Wed May  5 16:43:10 2021

@author: holge
"""

import dask.array as da
import numpy as np

if __name__ == "__main__":
    x1 = da.random.random((5000, 5000), chunks=(1000, 1000))
    x2 = da.random.random((5000, 5000), chunks=(1000, 1000))
    x1x2 = da.matmul(x1, x2)
    y = x1x2.compute()
    y2 = np.matmul(np.asarray(x1), np.asarray(x1))
    
    print(np.allclose(y,y2))
