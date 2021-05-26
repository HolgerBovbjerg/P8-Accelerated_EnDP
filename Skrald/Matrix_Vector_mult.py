import os

import numpy as np
import scipy.fft as scfft
from dask.distributed import Client, wait
import torch

import function_file as ff


def Naive_Mult(A, B):
    # dimension check
    if A.shape[1] == B.shape[0]:
        result = np.zeros((A.shape[0], B.shape[1]))
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(B)):
                    result[i][j] += A[i][k] * B[k][j]

        return result

    else:
        return print('Dimensions must match')



def DASK_batch_mult(matrix_input, vector_input, workers, batch_size, input_size, output_channels):
    client = Client(n_workers=workers)
    results = []
    batch_no = matrix_input.shape[0] // batch_size

    for i in range(batch_no):
        batch = client.scatter(matrix_input[i * batch_size: i * batch_size + batch_size])
        results.append(
            client.submit(ff.convolution_mean, batch, vector_input, batch_size, vector_input.shape[0])
        )

    client.gather(results)
    out_tensor = np.zeros((batch_size * batch_no, output_channels, input_size, input_size))
    for i in range(batch_no):
        out_tensor[i * batch_size: i * batch_size + batch_size] = results[i].result().reshape(batch_size,
                                                                                              output_channels,
                                                                                              input_size, input_size)

    return out_tensor


def DASK_panel_mult(matrix, vector, workers, input_size):
    """
    Calculates the matrix-vector product using numpy on n workers in parallel.
    The input matrix is divided into panels of

    """
    matrix = np.array(matrix[0]).transpose()
    client = Client(n_workers=workers)
    results = []
    blockno = matrix.shape[0] // input_size

    for block in range(blockno):
        # Send the data to the cluster as this is best practice for large data.
        data = matrix[input_size * block:input_size * block + input_size]
        big_future = client.scatter(data)
        results.append(
            client.submit(np.matmul, big_future, vector)
        )

    client.gather(results)
    out_matrix = np.vstack([result.result() for result in results])
    wait(out_matrix)
    client.close()

    return torch.tensor(out_matrix).view(1, 224, 224)


def fft_mult(photo_matrix, vector):
    '''
    This is the naive implementation of multiplication of circulant matrix with a vector, using the FFT.
    Each row of the photo and corresponding slice of the vector are FFT'ed in a for-loop
    Likewise the output is calculated piecewise in a double for-loop.

    :param photo_matrix: The nxn-shaped zeropadded input photo
    :param vector: The mean-vector of the kernel
    :return: The resulting mean-vector from the matrix-vector multiplication.
    '''
    n = photo_matrix.shape[0]
    ts = np.empty((n, n), dtype=np.complex_)
    ws = np.empty((n, n), dtype=np.complex_)
    for i in range(n):
        ts[i] = scfft.fft(photo_matrix[i])  # Lav row-wise fft hvis muligt
        ws[i] = scfft.fft(vector.reshape((n, n))[i])  # Pre-calculate q.reshape

    out = np.empty((n * n))
    for j in np.arange(n):
        x = 0
        shifted_ts = np.roll(ts, -j, 0)
        for i in range(n):
            x = x + (shifted_ts[-i] * ws[i])
        out[(j * n):(j + 1) * n] = np.real(scfft.ifft(x))

    return out


def fft_mult_vec(photo_matrix, vector):
    '''
    This is the first step of vectorizing the FFT multiplication algorithm.
    Here the entire photo_matrix and the vector are FFT'ed in one go.
    The output is still calculated with a double for-loop.

    :param photo_matrix: The nxn-shaped zeropadded input photo
    :param vector: The mean-vector of the kernel
    :return: The resulting mean-vector from the matrix-vector multiplication.
    '''
    n = photo_matrix.shape[0]
    ts = scfft.fft(photo_matrix, axis=1)
    ws = scfft.fft(vector.reshape((n, n)), axis=1)

    out = np.empty((n * n))
    for j in np.arange(n):
        x = 0
        shifted_ts = np.roll(ts, -j, 0)
        for i in range(n):
            x = x + (shifted_ts[-i] * ws[i])
        out[(j * n):(j + 1) * n] = np.real(scfft.ifft(x))

    return out


def fft_mult_vec_2(photo_matrix, vector):
    '''
    This is the next step in vectorizing the FFT-multiplication algorithm.
    Here the the output is calculated in a single for-loop, with a multiplication with ws in each iteration.

    TODO: Se om det giver bedre performance at bruge einsum fremfor np.sum. Måske lav det conditioned på størrelse af n

    :param photo_matrix: The nxn-shaped zeropadded input photo
    :param vector: The mean-vector of the kernel
    :return: The resulting mean-vector from the matrix-vector multiplication.
    '''
    n = photo_matrix.shape[0]
    ts = scfft.fft(photo_matrix, axis=1)
    ws = scfft.fft(vector.reshape((n, n)), axis=1)

    out = np.empty((n * n))
    for j in np.arange(n):
        x = np.sum(np.roll(np.flipud(np.roll(ts, -j, 0)), 1, 0) * ws, 0)
        out[(j * n):(j + 1) * n] = np.real(scfft.ifft(x))

    return out


def fft_mult_vec_3(photo_matrix, vector):
    '''
    This is the third step in the vectorization of the FFT-multiplication algorithm
    Here the multiplication with ws is moved outside the for-loop and done only once.

    :param photo_matrix: The nxn-shaped zeropadded input photo
    :param vector: The mean-vector of the kernel
    :return: The resulting mean-vector from the matrix-vector multiplication.
    '''
    n = photo_matrix.shape[0]
    ts = scfft.fft(photo_matrix, axis=1)
    ws = scfft.fft(vector.reshape((n, n)), axis=1)

    t_temp = np.empty((n * n, n), dtype=complex)
    for j in np.arange(n):
        t_temp[j * n:(j + 1) * n] = np.roll(np.flipud(np.roll(ts, -j, 0)), 1, 0)

    x = np.sum(t_temp.reshape(n, n, n) * ws, 1)
    out = np.real(scfft.ifft(x)).reshape(n ** 2)
    return out


def fft_mult_vec_4(photo_matrix, vector):
    '''

    :param photo_matrix: The nxn-shaped zeropadded input photo
    :param vector: The mean-vector of the kernel
    :return: The resulting mean-vector from the matrix-vector multiplication.
    '''
    n = photo_matrix.shape[0]
    ts = scfft.fft(photo_matrix, axis=1)
    ws = scfft.fft(vector.reshape((n, n)), axis=1)

    t_temp = np.empty((n * n, n), dtype=complex)
    for j in np.arange(n):
        t_temp[j * n:(j + 1) * n] = np.roll(np.flipud(np.roll(ts, -j, 0)), 1, 0)

    x = np.einsum('ij,kij->kj', ws, t_temp.reshape(n, n, n))
    out = np.real(scfft.ifft(x)).reshape(n ** 2)
    return out
