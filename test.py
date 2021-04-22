import numpy as np

A = np.array([
    [1, 2, 3],
    [1, 2, 3],
    [1, 2, 3]
])

B = np.transpose(A)+ np.ones(3)

C = np.sum(B*A, 1)
