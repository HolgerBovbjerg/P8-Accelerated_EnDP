import os

slow = True
# slow = False
if slow:
    os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import time

a = np.random.random((4000, 4000))
b = np.random.random((4000, 4000))
n = 20

print(f'starting with slow = {slow}')
start = time.time()
for i in range(n):
    c = np.matmul(a, b)

slut = time.time() - start
print(f'time = {slut}')
