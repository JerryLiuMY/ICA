from global_settings import DATA_PATH
import pickle5 as pickle
import numpy as np
import os


# model
batch_size = 128
vae_dict = {"epochs": 200, "lr": 0.001, "beta": 1}
mle_dict = {"epochs": 50, "lr": 0.001}
exp_dict = {"train_size": 10000, "valid_size": 2000, "simu_size": 5000}

# numerical integration
mc = 1000
min_lim, max_lim, space = -2.5, 2.5, 51


# experiments
num_trials = 25
num_lin = 15
m_n_dict = {
    "vae": [[2, 500], [2, 100]],
    "mleauto": [[2, 20], [2, 20]],
    "mleagd": [[2, 20], [2, 20]]
}


def save_params(m, n, seed):
    """ save parameters for neural network
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param seed: random seed for generating the variables
    :return:
    """

    sigma = 1.
    np.random.seed(seed)
    w = 3 * np.random.rand(n, m)
    b = 1 * np.random.rand(n)
    params = {"sigma": sigma, "w": w, "b": b}

    with open(os.path.join(DATA_PATH, f"params_{m}_{n}.pkl"), "wb") as handle:
        pickle.dump(params, handle)


def get_m_n(model_name):
    """ get m_n_li and m_li_n for experiments
    :param model_name: name of the model to run
    :return m_n_li: m plus n_li
    :return m_li_n: m_li plus n
    """

    [[n_min, n_max], [m_min, m_max]] = m_n_dict[model_name]
    m_n_li = [[n_min], list(np.round(np.exp(np.linspace(np.log(n_min), np.log(n_max), num_lin))).astype(int))]
    m_li_n = [list(np.round(np.exp(np.linspace(np.log(m_min), np.log(m_max), num_lin))).astype(int)), [m_max]]

    return m_n_li, m_li_n
