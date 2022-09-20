from global_settings import DATA_PATH
from params.params import save_params
from datetime import datetime
import pickle5 as pickle
import numpy as np
import pandas as pd
import torch
import os
import re


def generate_data(m, n, activation, size, seed=0):
    """ Generate data for non-linear ICA
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param activation: activation function for mlp
    :param size: number of samples to generate
    :param seed: random seed for dgp
    :return: dataframe of z and x
    """

    # load parameters
    activation_name = ''.join([_ for _ in re.sub("[\(\[].*?[\)\]]", "", str(activation)) if _.isalpha()])
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Building data with m={m}, n={n} "
          f"with activation={activation_name}")

    params_path = os.path.join(DATA_PATH, f"params_{m}_{n}")
    if not os.path.isdir(params_path):
        os.mkdir(params_path)

    params_file = os.path.join(params_path, f"seed_{seed}.pkl")
    if not os.path.isfile(params_file):
        save_params(m, n, seed)

    with open(params_file, "rb") as handle:
        params = pickle.load(handle)
        sigma = params["sigma"]
        w = params["w"]
        b = params["b"]

    # generate z and x
    np.random.seed(seed)
    z = np.empty(shape=(0, m))
    x = np.empty(shape=(0, n))

    for _ in range(size):
        z_ = np.random.multivariate_normal(mean=np.zeros(m), cov=np.eye(m))
        f_ = activation(torch.tensor(w @ z_ + b)).numpy()
        x_ = np.random.multivariate_normal(mean=f_, cov=(sigma**2)*np.eye(n))
        z = np.concatenate([z, z_.reshape(1, -1)], axis=0)
        x = np.concatenate([x, x_.reshape(1, -1)], axis=0)

    # training dataframe
    x_dict = {f"x{i}": x[:, i].reshape(-1) for i in range(x.shape[1])}
    z_dict = {f"z{i}": z[:, i].reshape(-1) for i in range(z.shape[1])}
    data_dict = {**x_dict, **z_dict}
    data_df = pd.DataFrame(data_dict)

    return data_df
