import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from torchvision import datasets, transforms

# import helper
import Matrix_Vector_mult as mv
import function_file as ff

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    imagepath = dir_path + "/Resized_images"
    im1 = ff.import_image(imagepath, 1)


    sobel_left = np.array([
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0]
    ], dtype='float32')


    # Convert the sobel operator into a convolution kernel adapted to the convolution operation
    sobel_left = sobel_left.reshape((1, 1, 3, 3))
    # Convolution output channel, here I set it to 3
    sobel_left = np.repeat(sobel_left, 3, axis=1)
    # Make into tensor and "reshape" into one vector
    sobel_left = torch.tensor(sobel_left).view(1, 1, 1, 27)

    sobel_top = np.array([
        [-1.0, -2.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0]
    ], dtype='float32')

    # Convert the sobel operator into a convolution kernel adapted to the convolution operation
    sobel_top = sobel_top.reshape((1, 1, 3, 3))
    # Convolution output channel, here I set it to 3
    sobel_top = np.repeat(sobel_top, 3, axis=1)
    # Make into tensor and "reshape" into one vector
    sobel_top = torch.tensor(sobel_top).view(1, 1, 1, 27)

    sobel_right = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0]
    ], dtype='float32')

    # Convert the sobel operator into a convolution kernel adapted to the convolution operation
    sobel_right = sobel_right.reshape((1, 1, 3, 3))
    # Convolution output channel, here I set it to 3
    sobel_right = np.repeat(sobel_right, 3, axis=1)
    # Make into tensor and "reshape" into one vector
    sobel_right = torch.tensor(sobel_right).view(1, 1, 1, 27)

    duo = torch.cat((sobel_left, sobel_right), dim=0)
    randfilt = torch.randn(64, 1, 1, 27)

    unfold = torch.nn.Unfold(kernel_size=(3, 3), padding=1)
    fold = torch.nn.Fold(output_size=(224, 224), kernel_size=(1, 1))
    im1_unf = unfold(im1)

    # -----------------------------------------------
    # We do this smart
    out_unf = torch.matmul(sobel_right, im1_unf)
    out_unf_dask = mv.DASK_mult(im1_unf, np.array(sobel_right[0][0]).transpose().reshape((27,)), 12, 224)
    out_duo_unf = torch.matmul(duo, im1_unf)
    out_rand_unf = torch.matmul(randfilt, im1_unf)
    # -----------------------------------------------

    # out = fold(out_unf)
    out = out_unf.view(1, 224, 224)
    out_duo = out_duo_unf.view(2, 224, 224)
    out_rand = out_rand_unf.view(64, 224, 224)
    out_dask = out_unf_dask.view(1, 224, 224)

    plt.imshow(out.permute(1, 2, 0), cmap='gray')
    plt.show()

    plt.imshow(out_dask.permute(1, 2, 0), cmap='gray')
    plt.show()

    # plt.imshow(out_rand[0], cmap='gray')
    # plt.show()
