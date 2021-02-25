import numpy as np
from scipy.linalg import circulant
import scipy.fft as scfft


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


# def FFT_mult(a, b):


n = 4
w = np.arange(n) + 1
W = circulant(w)
Block = np.tile(W, (n, n))

a = np.random.rand(n)
b = np.random.rand(n)
c = np.random.rand(n)
d = np.random.rand(n)
data = np.vstack([a, b, c, d])

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

# Q = np.random.rand(n*n)
Q = np.arange(n * n) + 1




# test = np.array([344, 352, 344, 320, 344, 352, 344, 320, 344, 352, 344, 320, 344, 352, 344, 320])
ts = np.zeros((n, n), dtype=np.complex_)
ws = np.zeros((n, n), dtype=np.complex_)
for i in range(n):
    ts[i] = scfft.fft(np.transpose(data[i]))
    ws[i] = scfft.fft(Q.reshape((n, n))[i])

out = np.zeros((n * n))
for j in np.arange(n):
    x = 0
    indices = np.arange(n)+j
    wrap = ts.take(indices, mode='wrap')
    for i in range(n):
        x = x + (wrap[i] * ws[i])
    out[(j * n):(j + 1) * n] = np.transpose(scfft.ifft(x))
    out = np.transpose(out)

# out((j * n) - n + 1: (j * n)) = ifft(x)
test = np.dot(data_block, Q)

print(test-out)
