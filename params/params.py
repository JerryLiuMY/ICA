from global_settings import DATA_PATH
import pickle5 as pickle
import numpy as np
import os

batch_size = 128
vae_dict = {"epochs": 200, "lr": 0.001, "beta": 1}
mle_dict = {"epochs": 200, "lr": 0.001}
exp_dict = {"train_size": 10000, "valid_size": 2000, "simu_size": 5000}
min_lim, max_lim, space = -2.5, 2.5, 51
mc = 1000


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

    with open(os.path.join(DATA_PATH, f"params_{m}_{n}.pkl"), "wb") as handle:
        pickle.dump(params, handle)
