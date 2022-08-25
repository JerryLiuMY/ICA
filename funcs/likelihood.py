from torch.distributions import MultivariateNormal
from funcs.normal import get_norm_lp
from global_settings import device
from params.params import mc
from params.params import min_lim, max_lim, space
import itertools
import numpy as np
import torch


def get_llh_mc(m, n, input_batch, model):
    """ Find log-likelihood from data and trained model
    :param m: latent dimension
    :param n: observed dimension
    :param input_batch: inputs related to observation [batch_size x dimension]
    :param model: trained model
    :return: log-likelihood
    """

    # define input
    x_batch, logs2_batch = input_batch
    x, s2 = x_batch, logs2_batch.exp()
    batch_size = x_batch.shape[0]

    # get reconstruction -- batch_size x mc x n
    mu_mc = torch.zeros(m).repeat(mc, 1).reshape(mc, m)
    mu_mc = mu_mc.repeat(batch_size, 1, 1).reshape(batch_size, mc, m)
    var_tril_mc = torch.eye(m).repeat(mc, 1, 1).reshape(mc, m, m)
    var_tril_mc = var_tril_mc.repeat(batch_size, 1, 1, 1).reshape(batch_size, mc, m, m)
    sampler = MultivariateNormal(loc=mu_mc, scale_tril=var_tril_mc)
    z_mc = sampler.sample()
    z_mc = z_mc.to(device)
    x_recon = model.decoder(z_mc)[0]

    # get covariance -- batch_size x mc x n x n
    s2_sqrt = s2.sqrt()
    s2_sqrt = s2_sqrt.repeat(1, n * n).reshape(batch_size, n, n)
    s2_sqrt = s2_sqrt.repeat(1, mc, 1, 1).reshape(batch_size, mc, n, n)
    eye = torch.eye(n).repeat(mc, 1, 1).reshape(mc, n, n)
    eye = eye.repeat(batch_size, 1, 1, 1).reshape(batch_size, mc, n, n)
    eye = eye.to(device)
    s2_cov_tril = s2_sqrt * eye

    # get input x -- batch_size x mc x n
    x = x.repeat(1, mc).reshape(batch_size, mc, n)

    # perform numerical integration
    llh = get_norm_lp(x, loc=x_recon, cov_tril=s2_cov_tril)
    llh = llh.to(torch.float64)
    llh_sample = llh.exp().sum(dim=1).log()
    llh_sample = torch.nan_to_num(llh_sample, neginf=np.log(torch.finfo(torch.float64).tiny))
    llh_batch = llh_sample.sum(dim=0)
    llh_batch = llh_batch.cpu().detach().numpy().tolist()

    return llh_batch


def get_llh_grid(m, n, input_batch, model):
    """ Find log-likelihood from data and trained model
    :param m: latent dimension
    :param n: observed dimension
    :param input_batch: inputs related to observation [batch_size x dimension]
    :param model: trained model
    :return: log-likelihood
    """

    # define input
    x_batch, logs2_batch = input_batch
    x, s2 = x_batch, logs2_batch.exp()
    batch_size = x_batch.shape[0]

    # prepare for numerical integration
    lin_space = np.linspace(min_lim, max_lim, space)
    grid_space = np.array([0.5 * (lin_space[i] + lin_space[i + 1]) for i in range(len(lin_space) - 1)])
    volume = ((max_lim - min_lim) / (space - 1)) ** m
    volume = torch.tensor(volume)

    # get reconstruction -- batch_size x grid_size x n
    z_grid = itertools.product(*[list(grid_space) for _ in range(m)])
    z_grid = np.array(list(z_grid)).astype(np.float32)
    z_grid = torch.tensor(z_grid)
    grid_size = z_grid.shape[0]
    z_grid = z_grid.repeat(batch_size, 1, 1).reshape(batch_size, grid_size, m)
    z_grid = z_grid.to(device)
    x_recon = model.decoder(z_grid)[0]

    # get covariance -- batch_size x grid_size x n x n
    s2_sqrt = s2.sqrt()
    s2_sqrt = s2_sqrt.repeat(1, n * n).reshape(batch_size, n, n)
    s2_sqrt = s2_sqrt.repeat(1, grid_size, 1, 1).reshape(batch_size, grid_size, n, n)
    eye = torch.eye(n).repeat(grid_size, 1, 1).reshape(grid_size, n, n)
    eye = eye.repeat(batch_size, 1, 1, 1).reshape(batch_size, grid_size, n, n)
    eye = eye.to(device)
    s2_cov_tril = s2_sqrt * eye

    # get input x -- batch_size x grid_size x n
    x = x.repeat(1, grid_size).reshape(batch_size, grid_size, n)

    # perform numerical integration
    log_prob_1 = get_norm_lp(x, loc=x_recon, cov_tril=s2_cov_tril)
    log_prob_2 = get_norm_lp(z_grid, loc=torch.zeros(z_grid.shape[-1]), cov_tril=torch.eye(z_grid.shape[-1]))
    log_prob_3 = torch.log(volume)
    llh = log_prob_1 + log_prob_2 + log_prob_3
    llh = llh.to(torch.float64)
    llh_sample = llh.exp().sum(dim=1).log()
    llh_sample = torch.nan_to_num(llh_sample, neginf=np.log(torch.finfo(torch.float64).tiny))
    llh_batch = llh_sample.sum(dim=0)
    llh_batch = llh_batch.cpu().detach().numpy().tolist()

    return llh_batch
