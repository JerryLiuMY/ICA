from global_settings import DATA_PATH
from scipy.stats import multivariate_normal
import pickle5 as pickle
import numpy as np
from emcee import EnsembleSampler
import torch
import os


def simu_post(x, m, n, activation):
    """ Get latent variables from MCMC
    :param x: observed variable x
    :param m: dimension of the latent variable
    :param n:dimension of the observed variable
    :param activation: activation function for mlp
    :return:
    """

    sampler = EnsembleSampler(nwalkers=2*m, ndim=m, log_prob_fn=get_log_prob, threads=8, args=(x, m, n, activation))
    p0 = np.random.randn(2 * m, m)
    sampler.run_mcmc(p0, skip_initial_state_check=True, nsteps=500)
    post = sampler.get_last_sample().coords[0, :]

    return post


def get_log_prob(z, x, m, n, activation):
    """ Get log-probability of the joint
    :param z: latent variable z
    :param x: observed variable x
    :param m: dimension of the latent variable
    :param n: dimension of the observed variable
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
    log_prob_1 = multivariate_normal.logpdf(x, mean=f, cov=(sigma**2)*np.eye(n))
    log_prob_2 = multivariate_normal.logpdf(z, mean=np.zeros(m), cov=np.eye(m))
    log_prob = log_prob_1 + log_prob_2

    return log_prob
