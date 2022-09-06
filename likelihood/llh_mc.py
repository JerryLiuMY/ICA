from torch.distributions import MultivariateNormal
from likelihood.dist import get_normal_lp
from global_settings import device
from params.params import mc
import numpy as np
import torch


def initialize_mc(m, n, x, model, logs2):
    """ Initialize monte-carlo integration
    :param m: latent dimension
    :param n: observed dimension
    :param x: inputs related to the observation x data
    :param model: trained model
    :param logs2: log of the estimated s2
    :return: x, x_recon, s2_cov_tril
    """

    # define input
    data_size = x.size(dim=0)

    # get reconstruction -- data_size x mc x n
    mu_mc = torch.zeros(m).repeat(mc, 1).reshape(mc, m)
    mu_mc = mu_mc.repeat(data_size, 1, 1).reshape(data_size, mc, m)
    var_tril_mc = torch.eye(m).repeat(mc, 1, 1).reshape(mc, m, m)
    var_tril_mc = var_tril_mc.repeat(data_size, 1, 1, 1).reshape(data_size, mc, m, m)
    sampler = MultivariateNormal(loc=mu_mc, scale_tril=var_tril_mc)
    z_mc = sampler.sample()
    z_mc = z_mc.to(device)
    x_recon = model.decoder(z_mc)[0]

    # get covariance -- data_size x mc x n x n
    s2_sqrt = logs2.exp().sqrt()
    s2_sqrt = s2_sqrt.repeat(1, n * n).reshape(data_size, n, n)
    s2_sqrt = s2_sqrt.repeat(1, mc, 1, 1).reshape(data_size, mc, n, n)
    eye = torch.eye(n).repeat(mc, 1, 1).reshape(mc, n, n)
    eye = eye.repeat(data_size, 1, 1, 1).reshape(data_size, mc, n, n)
    eye = eye.to(device)
    s2_cov_tril = s2_sqrt * eye

    # get input x -- data_size x mc x n
    x = x.repeat(1, mc).reshape(data_size, mc, n)

    return x, x_recon, s2_cov_tril


def get_llh_mc(m, n, x, model, logs2):
    """ Find log-likelihood from data and trained model
    :param m: latent dimension
    :param n: observed dimension
    :param x: inputs related to the observation x data
    :param model: trained model
    :param logs2: log of the estimated s2
    :return: log-likelihood
    """

    # perform numerical integration
    x, x_recon, s2_cov_tril = initialize_mc(m, n, x, model, logs2)
    llh = get_normal_lp(x, loc=x_recon, cov_tril=s2_cov_tril)
    llh = llh.to(torch.float64)
    llh_sample = llh.exp().sum(dim=1).log()
    llh_sample = torch.nan_to_num(llh_sample, neginf=np.log(torch.finfo(torch.float64).tiny))

    return llh_sample


def get_grad_mc(m, n, x, model, logs2):
    """ Find log-likelihood from data and trained model
    :param m: latent dimension
    :param n: observed dimension
    :param x: inputs related to the observation x data
    :param model: trained model
    :param logs2: log of the estimated s2
    :return: log-likelihood
    """

    pass
