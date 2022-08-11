from global_settings import DATA_PATH
from params.params import save_params
from datetime import datetime
import pickle5 as pickle
import numpy as np
import pandas as pd
import torch
import os


def generate_data(m, n, activation, train_size, valid_size):
    """ Generate data for non-linear ICA
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param activation: activation function for mlp
    :param train_size: number of samples for training set
    :param valid_size: number of samples for validation set
    :return: dataframe of z and x
    """

    # load parameters
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Building data with m={m}, n={n} "
          f"with activation={''.join([_ for _ in str(activation) if _.isalpha()])}")

    params_path = os.path.join(DATA_PATH, f"params_{m}_{n}.pkl")
    if not os.path.isfile(params_path):
        save_params(m, n)

    with open(params_path, "rb") as handle:
        params = pickle.load(handle)
        sigma = params["sigma"]
        w = params["w"]
        b = params["b"]

    # generate z and x
    np.random.seed(10)
    z = np.empty(shape=(0, m))
    x = np.empty(shape=(0, n))

    for _ in range(train_size + valid_size):
        z_ = np.random.multivariate_normal(mean=np.zeros(m), cov=np.eye(m))
        f_ = activation(torch.tensor(w @ z_ + b)).numpy()
        x_ = np.random.multivariate_normal(mean=f_, cov=(sigma**2)*np.eye(n))
        z = np.concatenate([z, z_.reshape(1, -1)], axis=0)
        x = np.concatenate([x, x_.reshape(1, -1)], axis=0)

    # training dataframe
    x_train = x[:train_size, :]
    z_train = z[:train_size, :]
    x_train_dict = {f"x{i}": x_train[:, i].reshape(-1) for i in range(x_train.shape[1])}
    z_train_dict = {f"z{i}": z_train[:, i].reshape(-1) for i in range(z_train.shape[1])}
    train_dict = {**x_train_dict, **z_train_dict}
    train_df = pd.DataFrame(train_dict)

    # validation dataframe
    x_valid = x[train_size:, :]
    z_valid = z[train_size:, :]
    x_valid_dict = {f"x{i}": x_valid[:, i].reshape(-1) for i in range(x_valid.shape[1])}
    z_valid_dict = {f"z{i}": z_valid[:, i].reshape(-1) for i in range(z_valid.shape[1])}
    valid_dict = {**x_valid_dict, **z_valid_dict}
    valid_df = pd.DataFrame(valid_dict)

    return train_df, valid_df
