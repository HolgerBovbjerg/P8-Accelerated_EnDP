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



n = 4

a = np.random.randint(0, 255, n)
b = np.random.randint(0, 255, n)
c = np.random.randint(0, 255, n)
d = np.random.randint(0, 255, n)


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

Q = np.random.randint(0, 100, n*n)


ts = np.zeros((n, n), dtype=np.complex_)
ws = np.zeros((n, n), dtype=np.complex_)
for i in range(n):
    ts[i] = scfft.fft(data[i])
    ws[i] = scfft.fft(Q.reshape((n, n))[i])

out = np.zeros((n * n))
for j in np.arange(n):
    x = 0
    shifted_ts = np.roll(ts, -j, 0)
    for i in range(n):
        x = x + (shifted_ts[-i] * ws[i])
    out[(j * n):(j + 1) * n] = np.transpose(scfft.ifft(x))
    out = np.transpose(out)

test = np.dot(data_block, Q)
diff = test-out
print(diff)
