import numpy as np
from scipy import linalg



def cov_mult(conv_matrix, cov_matrix):
    conv_matrix = np.transpose(conv_matrix)
    return np.matmul(np.matmul(conv_matrix, cov_matrix), np.transpose(conv_matrix))


def relu(input_samples):
    # da.maximum(input_samples, 0, out= input_samples)  # This might be faster as it puts result in same variable.
    return np.maximum(input_samples, 0)


def mvn_random(mean, cov, N, dim):
    np.random.seed(10)
    epsilon = 0.0001
    A = linalg.cholesky(cov + epsilon * np.eye(dim), lower=True)
    z = np.random.standard_normal(size=(N, dim))
    x = np.outer(np.ones((N,)), mean).transpose() + np.dot(A, z.transpose())
    return x


def single_mean_single_covariance_validator(convmatrix, kernels, covariances, batch_size, total_kernels, input_size):
    out_image_list = []
    for i in range(batch_size):
        pre_relu_samples = []
        post_relu_samples = []
        end_means = []
        convolved_means = []
        convolved_covariances = []
        for j in range(total_kernels):
            convolved_covariances.append(
                cov_mult(convmatrix[i, :, :], covariances[j, :, :])
            )
            convolved_means.append(
                np.matmul(kernels[j, :], convmatrix[i, :, :])
            )
        for (mean, covariance) in zip(convolved_means, convolved_covariances):
            pre_relu_samples.append(
                # We could use client.map when we have futures in a list.
                mvn_random(mean, covariance, 1000000, input_size ** 2)
            )
        for samples in pre_relu_samples:
            post_relu_samples.append(
                relu(samples)
            )
        for samples in post_relu_samples:
            end_means.append(
                samples.mean(axis=1)
            )

        out_image_list.append(
            end_means
        )
    return out_image_list
