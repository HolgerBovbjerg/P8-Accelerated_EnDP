# -*- coding: utf-8 -*-
"""
Created on Mon May  3 13:35:51 2021

@author: holge
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import time
import torch

import function_file as ff
import Matrix_Vector_mult as mv


if __name__ == "__main__":
    #%% Import images
    dir_path = os.path.dirname(os.path.realpath(__file__))
    imagepath = dir_path + "/Resized_images"
    batch_size = 16
    convmatrix = np.empty((batch_size, 27, 224**2))
    
    start_time = time.time()
    
    for i in range(batch_size):
        image = ff.import_image(imagepath, i)
        convmatrix[i, :, :] = ff.image2convmatrix(image, 3, 1)
    
    #%% Convolution layer 1
    input_channels = 3
    input_size = 224
    input_kernels = 64
    kernel_size = 3
    output_channels = 64
    output_size = 224

    W1 = np.random.rand(input_kernels, kernel_size, kernel_size)
    # W1[63, :, :] = sobel_left
    W1_vec = np.empty((input_kernels, kernel_size**2*input_channels))
    for i in range(input_kernels):
        W1_vec[i, :] = ff.filt2vec(W1[i, :, :], input_channels)

    start = time.time()
    out1 = np.empty((batch_size, output_channels, input_size**2))
    for i in range(batch_size):
        for j in range(input_kernels):
            out1[i, j, :] = np.matmul(W1_vec[j, :], convmatrix[i, :, :])

    out_images1 = np.empty((batch_size, output_channels, output_size, output_size))
    for i in range(batch_size):
        for j in range(output_channels):
            out_images1[i, j, :, :] = ff.convout2image(out1[i, j, :], (output_size, output_size))

    execution_time_conv1 = (time.time() - start)
    
    
    plt.imshow(out_images1[1, 1, :, :], cmap='gray')
    plt.show()
    
    #%% ReLU
    # relu = torch.nn.ReLU()
    # input size = 32
    # sample_size = 1000
    # z = np.empty((batch_size, output_channels, input_size**2, sample_size))
    # for i in range(batch_size):
    #     for j in range(output_channels):
    #         z[i, j, :, :] = np.random.multivariate_normal(mu_conv1,
    #                                                       Sigma_conv1,
    #                                                       1000)
    # g = np.array(relu(torch.tensor(z)))
    # mu_g = mean(g)
    # Sigma_g = cov(g) 
    # ReLU_out1 = 
    
    #%% Convolution layer 2
    padsize = 1
    input_channels = 64
    input_size = 224
    input_kernels = 64
    kernel_size = 3
    output_channels = 64
    output_size = 224

    convmatrix2 = ff.image2convmatrix(torch.tensor(out_images1), kernel_size, padsize)
    
    W2 = np.random.rand(input_channels, kernel_size, kernel_size)
    W2_vec = np.empty((input_kernels, kernel_size**2*input_channels))
    for i in range(input_kernels):
        W2_vec[i, :] = ff.filt2vec(W2[i, :, :], input_channels)

    start = time.time()
    out2 = np.empty((batch_size, output_channels, input_size**2))
    for i in range(batch_size):
        for j in range(input_kernels):
            out2[i, j, :] = np.matmul(W2_vec[j, :], convmatrix2[i, :, :])

    out_images2 = np.empty((batch_size, output_channels, output_size, output_size))
    for i in range(batch_size):
        for j in range(output_channels):
            out_images2[i, j, :, :] = ff.convout2image(out2[i, j, :],
                                                       (output_size, output_size))

    execution_time_conv2 = (time.time() - start)

    plt.imshow(out_images2[1, 1, :, :], cmap='gray')
    plt.show()

    #%% max pool 1
    kernel_size = 2
    stride_size = 2
    
    maxpool_layer1 = torch.nn.MaxPool2d(kernel_size, stride=stride_size)
    start = time.time()
    maxpool_out1 = np.array(maxpool_layer1(torch.tensor(out_images1)))
    execution_time_maxpool1 = (time.time() - start)
    plt.imshow(maxpool_out1[1, 1, :, :], cmap='gray')
    plt.show()

    #%% Convolution layer 3
    padsize = 1
    input_channels = 64
    input_size = 112
    input_kernels = 64
    kernel_size = 3
    output_channels = 128
    output_size = 112

    convmatrix3 = ff.image2convmatrix(torch.tensor(maxpool_out1), kernel_size, padsize)
    W3 = np.random.rand(input_channels, kernel_size, kernel_size)
    W3_vec = np.empty((input_kernels, kernel_size**2*input_channels))
    for i in range(input_kernels):
        W3_vec[i, :] = ff.filt2vec(W3[i, :, :], input_channels)
        
    start = time.time()
    out3 = np.empty((batch_size, output_channels, input_size**2))
    for i in range(batch_size):
        for j in range(input_kernels):
            out3[i, j, :] = np.matmul(W3_vec[j, :], convmatrix3[i, :, :])

    out_images3 = np.empty((batch_size, output_channels, output_size, output_size))
    for i in range(batch_size):
        for j in range(output_channels):
            out_images3[i, j, :, :] = ff.convout2image(out3[i, j, :],
                                                       (output_size, output_size))

    execution_time_conv3 = (time.time() - start)

    plt.imshow(out_images3[1, 1, :, :], cmap='gray')
    plt.show()
    
    #%% Convolution layer 4
    padsize = 1
    input_channels = 128
    input_size = 112
    input_kernels = 128
    kernel_size = 3
    output_channels = 128
    output_size = 112

    convmatrix4 = ff.image2convmatrix(torch.tensor(out_images3), kernel_size, padsize)
    W4 = np.random.rand(input_channels, kernel_size, kernel_size)
    W4_vec = np.empty((input_kernels, kernel_size**2*input_channels))
    for i in range(input_kernels):
        W4_vec[i, :] = ff.filt2vec(W4[i, :, :], input_channels)
        
    start = time.time()
    out4 = np.empty((batch_size, output_channels, input_size**2))
    for i in range(batch_size):
        for j in range(input_kernels):
            out4[i, j, :] = np.matmul(W4_vec[j, :], convmatrix4[i, :, :])

    out_images4 = np.empty((batch_size, output_channels, output_size, output_size))
    for i in range(batch_size):
        for j in range(output_channels):
            out_images4[i, j, :, :] = ff.convout2image(out4[i, j, :],
                                                       (output_size, output_size))

    execution_time_conv4 = (time.time() - start)

    plt.imshow(out_images4[1, 1, :, :], cmap='gray')
    plt.show()
    
    #%% max pool 2
    kernel_size = 2
    stride_size = 2
    
    maxpool_layer2 = torch.nn.MaxPool2d(kernel_size, stride=stride_size)
    start = time.time()
    maxpool_out2 = np.array(maxpool_layer2(torch.tensor(out_images4)))
    execution_time_maxpool2 = (time.time() - start)
    plt.imshow(maxpool_out2[1, 1, :, :], cmap='gray')
    plt.show()
    
    #%% Convolution layer 5
    padsize = 1
    input_channels = 128
    input_size = 56
    input_kernels = 128
    kernel_size = 3
    output_channels = 256
    output_size = 56

    convmatrix5 = ff.image2convmatrix(torch.tensor(maxpool_out2), kernel_size, padsize)
    W5 = np.random.rand(input_channels, kernel_size, kernel_size)
    W5_vec = np.empty((input_kernels, kernel_size**2*input_channels))
    for i in range(input_kernels):
        W5_vec[i, :] = ff.filt2vec(W5[i, :, :], input_channels)
        
    start = time.time()
    out5 = np.empty((batch_size, output_channels, input_size**2))
    for i in range(batch_size):
        for j in range(input_kernels):
            out5[i, j, :] = np.matmul(W5_vec[j, :], convmatrix5[i, :, :])

    out_images5 = np.empty((batch_size, output_channels, output_size, output_size))
    for i in range(batch_size):
        for j in range(output_channels):
            out_images5[i, j, :, :] = ff.convout2image(out5[i, j, :],
                                                        (output_size, output_size))

    execution_time_conv5 = (time.time() - start)

    plt.imshow(out_images5[1, 1, :, :], cmap='gray')
    plt.show()
    
    #%% Convolution layer 6
    padsize = 1
    input_channels = 256
    input_size = 56
    input_kernels = 256
    kernel_size = 3
    output_channels = 256
    output_size = 56

    convmatrix6 = ff.image2convmatrix(torch.tensor(out_images5), kernel_size, padsize)
    W6 = np.random.rand(input_channels, kernel_size, kernel_size)
    W6_vec = np.empty((input_kernels, kernel_size**2*input_channels))
    for i in range(input_kernels):
        W6_vec[i, :] = ff.filt2vec(W6[i, :, :], input_channels)
        
    start = time.time()
    out6 = np.empty((batch_size, output_channels, input_size**2))
    for i in range(batch_size):
        for j in range(input_kernels):
            out6[i, j, :] = np.matmul(W6_vec[j, :], convmatrix6[i, :, :])

    out_images6 = np.empty((batch_size, output_channels, output_size, output_size))
    for i in range(batch_size):
        for j in range(output_channels):
            out_images6[i, j, :, :] = ff.convout2image(out6[i, j, :],
                                                        (output_size, output_size))

    execution_time_conv6 = (time.time() - start)

    plt.imshow(out_images6[1, 1, :, :], cmap='gray')
    plt.show()
    
    #%% Convolution layer 7
    padsize = 1
    input_channels = 256
    input_size = 56
    input_kernels = 256
    kernel_size = 3
    output_channels = 256
    output_size = 56

    convmatrix7 = ff.image2convmatrix(torch.tensor(out_images6), kernel_size, padsize)
    W7 = np.random.rand(input_channels, kernel_size, kernel_size)
    W7_vec = np.empty((input_kernels, kernel_size**2*input_channels))
    for i in range(input_kernels):
        W7_vec[i, :] = ff.filt2vec(W7[i, :, :], input_channels)
        
    start = time.time()
    out7 = np.empty((batch_size, output_channels, input_size**2))
    for i in range(batch_size):
        for j in range(input_kernels):
            out7[i, j, :] = np.matmul(W7_vec[j, :], convmatrix7[i, :, :])

    out_images7 = np.empty((batch_size, output_channels, output_size, output_size))
    for i in range(batch_size):
        for j in range(output_channels):
            out_images7[i, j, :, :] = ff.convout2image(out7[i, j, :],
                                                        (output_size, output_size))

    execution_time_conv7 = (time.time() - start)

    plt.imshow(out_images7[1, 1, :, :], cmap='gray')
    plt.show()
    
    #%% max pool 3
    kernel_size = 2
    stride_size = 2
    
    maxpool_layer3 = torch.nn.MaxPool2d(kernel_size, stride=stride_size)
    start = time.time()
    maxpool_out3 = np.array(maxpool_layer3(torch.tensor(out_images7)))
    execution_time_maxpool3 = (time.time() - start)
    plt.imshow(maxpool_out3[1, 1, :, :], cmap='gray')
    plt.show()
    
    #%% Convolution layer 8
    padsize = 1
    input_channels = 256
    input_size = 28
    input_kernels = 256
    kernel_size = 3
    output_channels = 512
    output_size = 28

    convmatrix8 = ff.image2convmatrix(torch.tensor(maxpool_out3), kernel_size, padsize)
    W8 = np.random.rand(input_channels, kernel_size, kernel_size)
    W8_vec = np.empty((input_kernels, kernel_size**2*input_channels))
    for i in range(input_kernels):
        W8_vec[i, :] = ff.filt2vec(W8[i, :, :], input_channels)
        
    start = time.time()
    out8 = np.empty((batch_size, output_channels, input_size**2))
    for i in range(batch_size):
        for j in range(input_kernels):
            out8[i, j, :] = np.matmul(W8_vec[j, :], convmatrix8[i, :, :])

    out_images8 = np.empty((batch_size, output_channels, output_size, output_size))
    for i in range(batch_size):
        for j in range(output_channels):
            out_images8[i, j, :, :] = ff.convout2image(out8[i, j, :],
                                                        (output_size, output_size))

    execution_time_conv8 = (time.time() - start)

    plt.imshow(out_images8[1, 1, :, :], cmap='gray')
    plt.show()
    
    #%% Convolution layer 9
    padsize = 1
    input_channels = 512
    input_size = 28
    input_kernels = 512
    kernel_size = 3
    output_channels = 512
    output_size = 28

    convmatrix9 = ff.image2convmatrix(torch.tensor(out_images8), kernel_size, padsize)
    W9 = np.random.rand(input_channels, kernel_size, kernel_size)
    W9_vec = np.empty((input_kernels, kernel_size**2*input_channels))
    for i in range(input_kernels):
        W9_vec[i, :] = ff.filt2vec(W9[i, :, :], input_channels)
        
    start = time.time()
    out9 = np.empty((batch_size, output_channels, input_size**2))
    for i in range(batch_size):
        for j in range(input_kernels):
            out9[i, j, :] = np.matmul(W9_vec[j, :], convmatrix9[i, :, :])

    out_images9 = np.empty((batch_size, output_channels, output_size, output_size))
    for i in range(batch_size):
        for j in range(output_channels):
            out_images9[i, j, :, :] = ff.convout2image(out9[i, j, :],
                                                        (output_size, output_size))

    execution_time_conv9 = (time.time() - start)

    plt.imshow(out_images9[1, 1, :, :], cmap='gray')
    plt.show()
    
    #%% Convolution layer 10
    padsize = 1
    input_channels = 512
    input_size = 28
    input_kernels = 512
    kernel_size = 3
    output_channels = 512
    output_size = 28

    convmatrix10 = ff.image2convmatrix(torch.tensor(out_images9), kernel_size, padsize)
    W10 = np.random.rand(input_channels, kernel_size, kernel_size)
    W10_vec = np.empty((input_kernels, kernel_size**2*input_channels))
    for i in range(input_kernels):
        W9_vec[i, :] = ff.filt2vec(W10[i, :, :], input_channels)
        
    start = time.time()
    out10 = np.empty((batch_size, output_channels, input_size**2))
    for i in range(batch_size):
        for j in range(input_kernels):
            out9[i, j, :] = np.matmul(W10_vec[j, :], convmatrix10[i, :, :])

    out_images10 = np.empty((batch_size, output_channels, output_size, output_size))
    for i in range(batch_size):
        for j in range(output_channels):
            out_images10[i, j, :, :] = ff.convout2image(out10[i, j, :],
                                                        (output_size, output_size))

    execution_time_conv10 = (time.time() - start)

    plt.imshow(out_images10[1, 1, :, :], cmap='gray')
    plt.show()
    
    #%% max pool 4
    kernel_size = 2
    stride_size = 2
    
    maxpool_layer4 = torch.nn.MaxPool2d(kernel_size, stride=stride_size)
    start = time.time()
    maxpool_out4 = np.array(maxpool_layer4(torch.tensor(out_images10)))
    execution_time_maxpool4 = (time.time() - start)
    plt.imshow(maxpool_out4[1, 1, :, :], cmap='gray')
    plt.show()
    
    #%% Convolution layer 11
    padsize = 1
    input_channels = 512
    input_size = 14
    input_kernels = 512
    kernel_size = 3
    output_channels = 512
    output_size = 14

    convmatrix11 = ff.image2convmatrix(torch.tensor(maxpool_out4), kernel_size, padsize)
    W11 = np.random.rand(input_channels, kernel_size, kernel_size)
    W11_vec = np.empty((input_kernels, kernel_size**2*input_channels))
    for i in range(input_kernels):
        W11_vec[i, :] = ff.filt2vec(W11[i, :, :], input_channels)
        
    start = time.time()
    out11 = np.empty((batch_size, output_channels, input_size**2))
    for i in range(batch_size):
        for j in range(input_kernels):
            out11[i, j, :] = np.matmul(W11_vec[j, :], convmatrix11[i, :, :])

    out_images11 = np.empty((batch_size, output_channels, output_size, output_size))
    for i in range(batch_size):
        for j in range(output_channels):
            out_images11[i, j, :, :] = ff.convout2image(out11[i, j, :],
                                                        (output_size, output_size))

    execution_time_conv11 = (time.time() - start)

    plt.imshow(out_images11[1, 1, :, :], cmap='gray')
    plt.show()
    
    #%% Convolution layer 12
    padsize = 1
    input_channels = 512
    input_size = 14
    input_kernels = 512
    kernel_size = 3
    output_channels = 512
    output_size = 14

    convmatrix12 = ff.image2convmatrix(torch.tensor(out_images11), kernel_size, padsize)
    W12 = np.random.rand(input_channels, kernel_size, kernel_size)
    W12_vec = np.empty((input_kernels, kernel_size**2*input_channels))
    for i in range(input_kernels):
        W12_vec[i, :] = ff.filt2vec(W12[i, :, :], input_channels)
        
    start = time.time()
    out12 = np.empty((batch_size, output_channels, input_size**2))
    for i in range(batch_size):
        for j in range(input_kernels):
            out11[i, j, :] = np.matmul(W12_vec[j, :], convmatrix12[i, :, :])

    out_images12 = np.empty((batch_size, output_channels, output_size, output_size))
    for i in range(batch_size):
        for j in range(output_channels):
            out_images12[i, j, :, :] = ff.convout2image(out12[i, j, :],
                                                        (output_size, output_size))

    execution_time_conv12 = (time.time() - start)

    plt.imshow(out_images12[1, 1, :, :], cmap='gray')
    plt.show()
    
    #%% Convolution layer 13
    padsize = 1
    input_channels = 512
    input_size = 14
    input_kernels = 512
    kernel_size = 3
    output_channels = 512
    output_size = 14

    convmatrix13 = ff.image2convmatrix(torch.tensor(out_images12), kernel_size, padsize)
    W13 = np.random.rand(input_channels, kernel_size, kernel_size)
    W13_vec = np.empty((input_kernels, kernel_size**2*input_channels))
    for i in range(input_kernels):
        W13_vec[i, :] = ff.filt2vec(W13[i, :, :], input_channels)
        
    start = time.time()
    out13 = np.empty((batch_size, output_channels, input_size**2))
    for i in range(batch_size):
        for j in range(input_kernels):
            out13[i, j, :] = np.matmul(W13_vec[j, :], convmatrix13[i, :, :])

    out_images13 = np.empty((batch_size, output_channels, output_size, output_size))
    for i in range(batch_size):
        for j in range(output_channels):
            out_images13[i, j, :, :] = ff.convout2image(out13[i, j, :],
                                                        (output_size, output_size))

    execution_time_conv13 = (time.time() - start)

    plt.imshow(out_images13[1, 1, :, :], cmap='gray')
    plt.show()
    
    #%% max pool 4
    kernel_size = 2
    stride_size = 2
    
    maxpool_layer5 = torch.nn.MaxPool2d(kernel_size, stride=stride_size)
    start = time.time()
    maxpool_out5 = np.array(maxpool_layer5(torch.tensor(out_images13)))
    execution_time_maxpool5 = (time.time() - start)
    plt.imshow(maxpool_out5[1, 1, :, :], cmap='gray')
    plt.show()
    
    
    
    total_time = time.time() - start_time
    
    