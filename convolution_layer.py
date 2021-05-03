# -*- coding: utf-8 -*-
"""
Created on Mon May  3 13:35:51 2021

@author: holge
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import function_file as ff
import Matrix_Vector_mult as mv
import time

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    imagepath = dir_path + "/Resized_images"
    convmatrix = np.empty((27, 224**2, 16))
    for i in range(16):
        image = ff.import_image(imagepath, i)
        convmatrix[:, :, i] = ff.image2convmatrix(image, 3, 1)

    sobel_left = np.array([
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0]
    ], dtype='float32')

    W = np.random.rand(3, 3, 64)
    W[:, :, 63] = sobel_left
    W_vec = np.empty((27, 64))
    for i in range(64):
        W_vec[:, i] = ff.filt2vec(W[:, :, i], 3)

    # start = time.time()


    # for i in range(64):
    #     out = mv.Naive_Mult(W_vec[:,i], convmatrix)

    # execution_time_naive = (time.time()-start)/10

    start = time.time()

    for i in range(16):
        for j in range(64):
            out = np.matmul(W_vec[:, j], convmatrix[:, :, i])
    
    execution_time_numpy = (time.time()-start)/10

    out_image = ff.convout2image(out, (224, 224))

    plt.imshow(out_image, cmap='gray')
    plt.show()
