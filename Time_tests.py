from Matrix_Vector_mult import fft_mult, fft_mult_vec, fft_mult_vec_2, fft_mult_vec_3, fft_mult_vec_4
import time
import dill  # Used for saving the workspace with dill.dump_session()
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as scfft
import scipy.signal as ss

repetitions = 1000
sizes = np.arange(10, 200)

ref_times = np.empty(sizes.shape)
vec_times = np.empty(sizes.shape)
vec2_times = np.empty(sizes.shape)
vec3_times = np.empty(sizes.shape)
vec4_times = np.empty(sizes.shape)
fft2_times = np.empty(sizes.shape)
# matmul_times = np.empty(sizes.shape)

j = 0
for n in sizes:
    print(n)
    data = np.random.randint(0, 255, (n, n))  # photo + zeropadding
    vector = np.random.randint(0, 100, n * n)  # mean vector of weights
    # block = np.random.randint(0, 255, (n**2, n**2))

    # start = time.time()
    # for i in range(repetitions):
    #     temp1 = fft_mult(data, vector)
    #
    # end = time.time()
    # tid = (end-start)/repetitions
    # ref_times[j] = tid
    #
    # start = time.time()
    # for i in range(repetitions):
    #     temp2 = fft_mult_vec(data, vector)
    #
    # vec_times[j] = (time.time()-start)/repetitions
    #
    # start = time.time()
    # for i in range(repetitions):
    #     temp3 = fft_mult_vec_2(data, vector)
    #
    # vec2_times[j] = (time.time()-start)/repetitions
    #
    # start = time.time()
    # for i in range(repetitions):
    #     temp4 = fft_mult_vec_3(data, vector)
    #
    # vec3_times[j] = (time.time()-start)/repetitions
    #
    # start = time.time()
    # for i in range(repetitions):
    #     temp5 = fft_mult_vec_4(data, vector)
    #
    # vec4_times[j] = (time.time()-start)/repetitions

    start = time.time()
    for i in range(n*n):
        temp6 = np.real(scfft.ifft2(scfft.fft2(data, (n, n)) * scfft.fft2(vector.reshape((n, n)), (n, n))))

    fft2_times[j] = (time.time()-start)/(n*n)


    # start = time.time()
    # for i in range(10):  # Because the matmul function is so consistent we use only 10 repetitions.
    #     temp = np.matmul(block, vector)
    #
    # matmul_times[j] = (time.time()-start)/10

    j += 1

#
# plt.plot(sizes, matmul_times, label='np.matmul implementation')
# plt.plot(sizes, ref_times, label='Basic implementation')
# plt.plot(sizes, vec_times, label='Vers. 1 of vectorized implementation')
# plt.plot(sizes, vec2_times, label='Vers. 2 of vectorized implementation')
# plt.plot(sizes, vec3_times, label='Vers. 3 of vectorized implementation')
# plt.plot(sizes, vec4_times, label='Vers. 4 of vectorized implementation')
plt.plot(sizes, fft2_times, label='FFT2 implementation')
plt.legend()
plt.xlabel('Size of input, N')
plt.ylabel('Average execution time, seconds')
plt.show()

#
# plt.plot(sizes, ref_times/ref_times, label='Basic implementation')
# # plt.plot(sizes, matmul_times/ref_times, label='np.matmul implementation')
# # plt.plot(sizes, vec_times/ref_times, label='Vers. 1 of vectorized implementation')
# plt.plot(sizes, vec2_times/ref_times, label='Vers. 2 of vectorized implementation')
# # plt.plot(sizes, vec3_times/ref_times, label='Vers. 3 of vectorized implementation')
# plt.plot(sizes, vec4_times/ref_times, label='Vers. 4 of vectorized implementation')
# plt.plot(sizes, fft2_times/ref_times, label='FFT2 implementation')
# plt.legend()
# plt.title(f'Average execution time, relative to ref. {repetitions} repetitions')
# plt.xlabel('Size of input, N')
# plt.ylabel('Average execution time, relative to ref')
# # plt.savefig('file.pdf')
# plt.show()



# Code-snippet used to save and load all workspace variables.
# filename = 'Time_test_100_reps_N170_250.pkl'
# dill.dump_session(filename)
#
# # and to load the session again:
# dill.load_session(filename)
