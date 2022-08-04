import numpy as np
from torch import nn


def generate(m, n, sigma, params, size):
    """ Generate data for non-linear ICA
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param sigma: standard deviation of the target variable
    :param params: parameters for the non-linearity
    :param size: number of samples
    :return: z and x
    """

    z = np.empty(shape=(0, m))
    x = np.empty(shape=(0, n))
    for _ in range(size):
        z_ = np.random.multivariate_normal(mean=np.zeros(m), cov=np.eye(m)).reshape(1, -1)
        f_ = nonlinearity(z, m, n, params)
        x_ = np.random.multivariate_normal(mean=f_, cov=(sigma**2)*np.eye(n)).reshape(1, -1)
        z = np.concatenate([z, z_], axis=0)
        x = np.concatenate([x, x_], axis=0)

    return z, x


def nonlinearity(z, m, n, params):
    """ Non-linear function for nonlinear ICA
    :param z: latent vector
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param params: parameters for the non-linearity
    :return:
    """

    activation = params["activation"]
    activation_func = getattr(nn, activation)

    # generate non-linearity
    np.random.seed(10)
    w = np.random.rand(n, m)
    b = np.random.rand(n)
    f = activation_func(w @ z + b)

    return f
