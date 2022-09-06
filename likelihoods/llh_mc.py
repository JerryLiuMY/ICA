from torch.distributions import MultivariateNormal
from likelihoods.dist import get_normal_lp
from global_settings import device
from params.params import mc
import numpy as np
import torch


def build_mc(m, n, x, model, logs2):
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

    return x, model, logs2, x_recon, s2_cov_tril


def get_llh_mc(m, n, x, model, logs2):
    """ Find log-likelihood from data and trained model [mc]
    :param m: latent dimension
    :param n: observed dimension
    :param x: inputs related to the observation x data
    :param model: trained model
    :param logs2: log of the estimated s2
    :return: log-likelihood
    """

    # perform numerical integration for llh
    x, model, logs2, x_recon, s2_cov_tril = build_mc(m, n, x, model, logs2)
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
    :return: gradients w.r.t model and s2
    """

    # perform numerical integration for likelihood
    x, model, logs2, x_recon, s2_cov_tril = build_mc(m, n, x, model, logs2)
    llh = get_normal_lp(x, loc=x_recon, cov_tril=s2_cov_tril)
    llh = llh.to(torch.float64)
    llh_sample = llh.exp().sum(dim=1).log()
    llh_sample = torch.nan_to_num(llh_sample, neginf=np.log(torch.finfo(torch.float64).tiny))

    # perform numerical integration for model parameters
    objective = - 0.5 * logs2.exp().pow(-1) * torch.norm(x - x_recon, p=2, dim=2)
    objective.backward(torch.ones_like(objective))
    for param in model.parameters():
        print(param.grad.view(-1))

    model_int = llh
    model_int = model_int.to(torch.float64)
    model_int_sample = model_int.exp().sum(dim=1)

    # perform numerical integration for s2
    s2_int = - logs2 + (x - x_recon) + llh
    s2_int = s2_int.to(torch.float64)
    s2_int_sample = s2_int.exp().sum(dim=1)

    # take the ratio and find the gradients
    model_grad_sample = model_int_sample / llh_sample.exp()
    s2_grad_sample = s2_int_sample / llh_sample.exp()

    return model_grad_sample, s2_grad_sample
