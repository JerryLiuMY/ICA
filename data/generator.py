import numpy as np
import pandas as pd
import torch


def generate_data(m, n, coeffs, activation, size):
    """ Generate data for non-linear ICA
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param coeffs: coefficients for data generation
    :param activation: activation function for mlp
    :param size: number of samples
    :return: z and x
    """

    sigma = coeffs["sigma"]
    w = coeffs["w"]
    b = coeffs["b"]

    z = np.empty(shape=(0, m))
    x = np.empty(shape=(0, n))

    np.random.seed(10)
    for _ in range(size):
        z_ = np.random.multivariate_normal(mean=np.zeros(m), cov=np.eye(m))
        f_ = activation(torch.tensor(w @ z_ + b)).numpy()
        x_ = np.random.multivariate_normal(mean=f_, cov=(sigma**2)*np.eye(n))
        z = np.concatenate([z, z_.reshape(1, -1)], axis=0)
        x = np.concatenate([x, x_.reshape(1, -1)], axis=0)

    x_dict = {f"x{i}": x[:, i].reshape(-1) for i in range(x.shape[1])}
    z_dict = {f"z{i}": z[:, i].reshape(-1) for i in range(z.shape[1])}
    df_dict = {**x_dict, **z_dict}
    data_df = pd.DataFrame(df_dict)

    return data_df
