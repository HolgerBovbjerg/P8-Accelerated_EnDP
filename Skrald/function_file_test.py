# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import function_file as ff
import time


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    imagepath = dir_path + "/Resized_images"

    im1 = ff.import_image(imagepath, 1)
    convmatrix = ff.image2convmatrix(im1, 3, 1)

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

    for i in range(64):
        out = np.matmul(W_vec[:, i], convmatrix)

    execution_time_numpy = (time.time()-start)/64
    
    

    out_image = ff.convout2image(out, (224, 224))

    plt.imshow(out_image, cmap='gray')
    plt.show()
