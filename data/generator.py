import numpy as np


def generate(m, n, sigma, func):
    """ Generate data for non-linear ICA
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param sigma: standard deviation of the target variable
    :param func: activation function
    :return: x and z
    """

    z = np.random.normal(loc=0.0, scale=1.0, size=m)
    x = np.random.normal(loc=func(z), scale=sigma, size=n)

    return x, z
