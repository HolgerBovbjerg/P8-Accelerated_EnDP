import numpy as np
from scipy.linalg import circulant
import scipy.fft as scfft
import time
from IPython import get_ipython

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


def fft_mult(photo_matrix, vector):
    n = photo_matrix.shape[0]
    ts = np.zeros((n, n), dtype=np.complex_)
    ws = np.zeros((n, n), dtype=np.complex_)
    for i in range(n):
        ts[i] = scfft.fft(photo_matrix[i])  # Lav row-wise fft hvis muligt
        ws[i] = scfft.fft(vector.reshape((n, n))[i])  # Pre-calculate q.reshape

    out = np.zeros((n * n))
    for j in np.arange(n):
        x = 0
        shifted_ts = np.roll(ts, -j, 0)
        for i in range(n):
            x = x + (shifted_ts[-i] * ws[i])
        out[(j * n):(j + 1) * n] = np.real(scfft.ifft(x))

    return out


def fft_mult_vec(photo_matrix, vector):
    n = photo_matrix.shape[0]
    ts = scfft.fft(data, axis=1)
    ws = scfft.fft(vector.reshape((n, n)), axis=1)

    out = np.zeros((n * n))
    for j in np.arange(n):
        x = 0
        shifted_ts = np.roll(ts, -j, 0)
        for i in range(n):
            x = x + (shifted_ts[-i] * ws[i])
        out[(j * n):(j + 1) * n] = np.real(scfft.ifft(x))

    return out


def fft_mult_vec_2(photo_matrix, vector):
    n = photo_matrix.shape[0]
    ts = scfft.fft(data, axis=1)
    ws = scfft.fft(vector.reshape((n, n)), axis=1)

    out = np.zeros((n * n))
    for j in np.arange(n):
        x = np.sum(np.roll(np.flipud(np.roll(ts, -j, 0)), 1, 0) * ws, 0)
        out[(j * n):(j + 1) * n] = np.real(scfft.ifft(x))

    return out


def fft_mult_vec_3(photo_matrix, vector):
    n = photo_matrix.shape[0]
    ts = scfft.fft(data, axis=1)
    ws = scfft.fft(vector.reshape((n, n)), axis=1)

    t_temp = np.roll(np.flipud(np.roll(ts, -0, 0)), 1, 0)
    for j in np.arange(n-1) + 1:
        t_temp = np.vstack([
            t_temp,
            np.roll(np.flipud(np.roll(ts, -j, 0)), 1, 0)
        ])

    x = np.sum(t_temp.reshape(n, n, n) * ws, 1)
    out = np.real(scfft.ifft(x)).reshape(n ** 2)
    return out


def fft_mult_vec_4(photo_matrix, vector):
    n = photo_matrix.shape[0]
    ts = scfft.fft(data, axis=1)
    ws = scfft.fft(vector.reshape((n, n)), axis=1)

    t_temp = np.roll(np.flipud(np.roll(ts, -0, 0)), 1, 0)
    for j in np.arange(n-1) + 1:
        t_temp = np.vstack([
            t_temp,
            np.roll(np.flipud(np.roll(ts, -j, 0)), 1, 0)
        ])

    x = np.einsum('ij,kij->kj', ws, t_temp.reshape(n, n, n))
    out = np.real(scfft.ifft(x)).reshape(n ** 2)
    return out



# Lav det så hver circulant produkt regnes distribueret.


ipython = get_ipython()
n = 28

a = np.random.randint(0, 255, n)
b = np.random.randint(0, 255, n)
c = np.random.randint(0, 255, n)
d = np.random.randint(0, 255, n)


# data = np.vstack([a, b, c, d])  # photo + zeropadding
data = np.random.randint(0, 255, (n, n))  # photo + zeropadding
Q = np.random.randint(0, 100, n*n)  # mean vector of weights

A = circulant(data[0])
B = circulant(data[1])
C = circulant(data[2])
D = circulant(data[3])

data_block = np.block([
    [A, D, C, B],
    [B, A, D, C],
    [C, B, A, D],
    [D, C, B, A]
])


# result = fft_mult_vec_4(data, Q)
# test = np.dot(data_block, Q)
# diff = test-result
# print(diff)


print('ref')
ipython.magic("timeit -n 10000 -r 20 fft_mult(data, Q)")
print('vec')
ipython.magic("timeit -n 10000 -r 20 fft_mult_vec(data, Q)")
print('vec2')
ipython.magic("timeit -n 10000 -r 20 fft_mult_vec_2(data, Q)")
print('vec3')
ipython.magic("timeit -n 10000 -r 20 fft_mult_vec_3(data, Q)")
print('vec4')
ipython.magic("timeit -n 10000 -r 20 fft_mult_vec_4(data, Q)")

# ipython.magic("timeit np.dot(data_block, Q)")
