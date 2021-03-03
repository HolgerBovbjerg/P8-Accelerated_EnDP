from Matrix_Vector_mult import fft_mult, fft_mult_vec, fft_mult_vec_2, fft_mult_vec_3, fft_mult_vec_4
import time
import numpy as np
import matplotlib.pyplot as plt

repetitions = 3000
sizes = np.arange(10, 50)

ref_times = np.empty(sizes.shape)
vec_times = np.empty(sizes.shape)
vec2_times = np.empty(sizes.shape)
vec3_times = np.empty(sizes.shape)
vec4_times = np.empty(sizes.shape)

j = 0
for n in sizes:
    print(n)
    data = np.random.randint(0, 255, (n, n))  # photo + zeropadding
    vector = np.random.randint(0, 100, n * n)  # mean vector of weights

    start = time.time()
    for i in range(repetitions):
        temp = fft_mult(data, vector)

    end = time.time()
    tid = (end-start)/repetitions
    ref_times[j] = tid

    start = time.time()
    for i in range(repetitions):
        temp = fft_mult_vec(data, vector)

    vec_times[j] = (time.time()-start)/repetitions

    start = time.time()
    for i in range(repetitions):
        temp = fft_mult_vec_2(data, vector)

    vec2_times[j] = (time.time()-start)/repetitions

    start = time.time()
    for i in range(repetitions):
        temp = fft_mult_vec_3(data, vector)

    vec3_times[j] = (time.time()-start)/repetitions

    start = time.time()
    for i in range(repetitions):
        temp = fft_mult_vec_4(data, vector)

    vec4_times[j] = (time.time()-start)/repetitions

    j += 1


plt.plot(sizes, ref_times, label='Ref')
plt.plot(sizes, vec_times, label='Vec')
plt.plot(sizes, vec2_times, label='Vec_2')
plt.plot(sizes, vec3_times, label='Vec_3')
plt.plot(sizes, vec4_times, label='Vec_4')
plt.legend()
plt.xlabel('Size of input, N')
plt.ylabel('Average execution time, seconds')
plt.show()


plt.plot(sizes, ref_times/ref_times, label='Ref')
plt.plot(sizes, vec_times/ref_times, label='Vec')
plt.plot(sizes, vec2_times/ref_times, label='Vec_2')
plt.plot(sizes, vec3_times/ref_times, label='Vec_3')
plt.plot(sizes, vec4_times/ref_times, label='Vec_4')
plt.legend()
plt.xlabel('Size of input, N')
plt.ylabel('Average execution time, relative to ref')
plt.show()