# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import function_file as ff

dir_path = os.path.dirname(os.path.realpath(__file__))
imagepath = dir_path + "/Resized_images"

im1 = ff.import_image(imagepath, 1)
convmatrix = ff.image2convmatrix(im1, 3, 1)

sobel_left = np.array([
    [-1.0, 0.0, 1.0],
    [-2.0, 0.0, 2.0],
    [-1.0, 0.0, 1.0]
], dtype='float32')

W_vec = ff.filt2vec(sobel_left, 3)

out = np.matmul(W_vec, convmatrix)

out_image = ff.convout2image(out, (224, 224))

plt.imshow(out_image, cmap='gray')
plt.show()
