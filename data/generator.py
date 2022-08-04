import numpy as np
import torch


def generate(m, n, coeffs, params, size):
    """ Generate data for non-linear ICA
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param coeffs: coefficients for data generation
    :param params: parameters for the non-linearity
    :param size: number of samples
    :return: z and x
    """

    sigma = coeffs["sigma"]
    z = np.empty(shape=(0, m))
    x = np.empty(shape=(0, n))
    for _ in range(size):
        z_ = np.random.multivariate_normal(mean=np.zeros(m), cov=np.eye(m)).reshape(1, -1)
        f_ = nonlinearity(z_, coeffs, params).reshape(-1)
        x_ = np.random.multivariate_normal(mean=f_, cov=(sigma**2)*np.eye(n)).reshape(1, -1)
        z = np.concatenate([z, z_], axis=0)
        x = np.concatenate([x, x_], axis=0)

    return z, x


def nonlinearity(z_, coeffs, params):
    """ Non-linear function for nonlinear ICA
    :param z_: latent vector
    :param coeffs: coefficients for data generation
    :param params: parameters for the non-linearity
    :return: output from non-linearity
    """

    # generate non-linearity
    activation = params["activation"]
    w, b = coeffs["w"], coeffs["b"]
    f = activation(torch.tensor(w @ z_ + b)).numpy()

    return f
