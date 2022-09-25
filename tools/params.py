from global_settings import DATA_PATH
import pickle5 as pickle
import numpy as np
import itertools
import os


# model
batch_size = 128
vae_dict = {"epochs": 200, "lr": 0.001, "beta": 1}
mle_dict = {"epochs": 50, "lr": 0.001}
exp_dict = {"train_size": 10000, "valid_size": 2000, "simu_size": 5000}

# numerical integration
sigma, mc = 1., 1000
min_lim, max_lim, space = -2.5, 2.5, 51

# experiments
num_trials = 25
num_lin = 15
m_n_dict = {
    "vae": [[2, 500], [2, 75]],
    "mleauto": [[2, 12], [2, 12]],
    "mleagd": [[2, 12], [2, 12]]
}


def save_params(m, n, seed):
    """ Save parameters for neural network
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param seed: random seed for generating the variables
    :return:
    """

    np.random.seed(seed)
    w = 3 * np.random.rand(n, m)
    b = 1 * np.random.rand(n)
    params = {"sigma": sigma, "w": w, "b": b}

    params_path = os.path.join(DATA_PATH, f"params_{m}_{n}")
    params_file = os.path.join(params_path, f"seed_{seed}.pkl")
    with open(params_file, "wb") as handle:
        pickle.dump(params, handle)


def get_li_m_n(model_name):
    """ Get m_n_li and m_li_n for experiments
    :param model_name: name of the model to run
    :return m_n_li: m plus n_li
    :return m_li_n: m_li plus n
    """

    [[n_min, n_max], [m_min, m_max]] = m_n_dict[model_name]
    if model_name == "vae":
        n_li = list(np.round(np.exp(np.linspace(np.log(n_min), np.log(n_max), num_lin))).astype(int))
        m_li = list(np.round(np.exp(np.linspace(np.log(m_min), np.log(m_max), num_lin))).astype(int))
    else:
        n_li = list(np.arange(n_min, n_max + 1).astype(int))
        m_li = list(np.arange(m_min, m_max + 1).astype(int))

    m_n_li = [[n_min], [*set(n_li)]]
    m_li_n = [[*set(m_li)], [m_max]]

    return m_n_li, m_li_n


def get_iter_m_n(m_n_li, m_li_n):
    """ Get the iterable iter_m_n of experiments
    :param m_n_li: name of the model to run
    :param m_li_n: m plus n_li
    """

    m_n_iter = sorted(list(itertools.product(*m_n_li)))
    m_iter_n = sorted(list(itertools.product(*m_li_n)))
    m_iter_n = [_ for _ in m_iter_n if _ not in m_n_iter]
    iter_m_n = m_n_iter + m_iter_n

    return iter_m_n
