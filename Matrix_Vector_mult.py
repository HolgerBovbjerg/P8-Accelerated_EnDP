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

    # t_temp = np.hstack([ts, np.roll(ts, 1, 0), np.roll(ts, 2, 0), np.roll(ts, 3, 0)])

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
        x = 0
        x = np.sum(np.roll(np.flipud(np.roll(ts, -j, 0)), 1, 0) * ws, 0)
        # shifted_ts = np.roll(ts, -j, 0)
        # for i in range(n):
        #     x = x + (shifted_ts[-i] * ws[i])
        out[(j * n):(j + 1) * n] = np.real(scfft.ifft(x))

    return out

ipython = get_ipython()
n = 4

a = np.random.randint(0, 255, n)
b = np.random.randint(0, 255, n)
c = np.random.randint(0, 255, n)
d = np.random.randint(0, 255, n)


data = np.vstack([a, b, c, d])  # photo + zeropadding
Q = np.random.randint(0, 100, n*n)  # mean vector of weights

A = circulant(a)
B = circulant(b)
C = circulant(c)
D = circulant(d)

data_block = np.block([
    [A, D, C, B],
    [B, A, D, C],
    [C, B, A, D],
    [D, C, B, A]
])


out = fft_mult_vec_2(data, Q)



test = np.dot(data_block, Q)
diff = test-out
print(diff)

# ipython.magic("timeit np.dot(data_block, Q)")

ipython.magic("timeit fft_mult_vec_2(data, Q)")
ipython.magic("timeit fft_mult_vec(data, Q)")
ipython.magic("timeit fft_mult(data, Q)")
