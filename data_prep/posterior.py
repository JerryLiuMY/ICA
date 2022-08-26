from global_settings import DATA_PATH
from scipy.stats import norm
import pickle5 as pickle
import numpy as np
import torch
import emcee
import os


def simu_post(x, m, n, activation, simu_size):
    """ Get latent variables from MCMC
    :param x: observed variable x
    :param m: dimension of the latent variable
    :param n:dimension of the observed variable
    :param activation: activation function for mlp
    :param simu_size: number of simulation samples to generate
    :return:
    """

    sampler = emcee.EnsembleSampler(simu_size, m, get_log_prob, args=[x, m, n, activation])
    p0 = np.random.randn(simu_size, m)
    post = sampler.run_mcmc(p0, nsteps=1000)[0]

    return post


def get_log_prob(z, x, m, n, activation):
    """ Get log-probability of the joint
    :param z: latent variable z
    :param x: observed variable x
    :param m: dimension of the latent variable
    :param n:dimension of the observed variable
    :param activation: activation function for mlp
    :return:
    """

    params_path = os.path.join(DATA_PATH, f"params_{m}_{n}.pkl")
    with open(params_path, "rb") as handle:
        params = pickle.load(handle)
        sigma = params["sigma"]
        w = params["w"]
        b = params["b"]

    f = activation(torch.tensor(w @ z + b)).numpy()
    log_prob_1 = norm.logpdf(x, loc=f, scale=sigma)
    log_prob_2 = norm.logpdf(z, loc=0, scale=1)
    log_prob = log_prob_1 + log_prob_2

    return log_prob
