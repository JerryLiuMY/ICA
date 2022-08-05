import numpy as np
import pickle5 as pickle


def save_params(m, n):
    """ save parameters for neural network
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :return:
    """

    sigma = 1
    np.random.seed(10)
    w = 3 * np.random.rand(n, m)
    b = 1 * np.random.rand(n)
    params = {"sigma": sigma, "w": w, "b": b}

    with open("")

