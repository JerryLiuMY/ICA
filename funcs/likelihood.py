from torch.distributions import MultivariateNormal
from global_settings import device
import itertools
import numpy as np
import torch


def get_llh_batch(m, n, input_batch, model):
    """ Find log-likelihood from data and trained model
    :param m: latent dimension
    :param n: observed dimension
    :param input_batch: inputs related to observation [batch_size x dimension]
    :param model: trained model
    :return: log-likelihood
    """

    # define input
    x_batch, mean_batch, logs2_batch = input_batch

    # prepare for numerical integration
    min_lim, max_lim, space = -2, 2, 21
    lin_space = np.linspace(min_lim, max_lim, space)
    grid_space = np.array([0.5 * (lin_space[i] + lin_space[i + 1]) for i in range(len(lin_space) - 1)])
    volume = ((max_lim - min_lim) / (space - 1)) ** m

    # get the tensors
    x, mean, logs2 = x_batch, mean_batch, logs2_batch
    s2 = logs2.exp()

    z_grid = itertools.product(*[list(grid_space) for _ in range(m)])
    z_grid = np.array(list(z_grid)).astype(np.float32)
    z_grid = torch.tensor(z_grid).to(device)
    x_recon = model.decoder(z_grid)[0]

    # reshape the tensors
    batch_size = x_batch.shape[0]
    grid_size = z_grid.shape[0]
    x_recon = x_recon.repeat(batch_size, 1).reshape(batch_size, grid_size, n)
    z_grid = z_grid.repeat(batch_size, 1).reshape(batch_size, grid_size, m)
    x = x.repeat(1, grid_size).reshape(batch_size, grid_size, n)
    s2 = s2.repeat(1, grid_size).reshape(batch_size, grid_size, 1)
    s2 = s2.repeat(1, 1, n * n).reshape(batch_size, grid_size, n, n)
    eye = torch.eye(n).repeat(batch_size * grid_size, 1).reshape(batch_size, grid_size, n, n)
    s2_cov = s2 * eye

    # perform numerical integration
    log_prob_1 = get_log_prob(x, x_recon, s2_cov)
    log_prob_2 = get_log_prob(z_grid, torch.zeros(z_grid.shape[-1]), torch.eye(z_grid.shape[-1]))
    llh = log_prob_1 + log_prob_2 + np.log(volume)
    llh_sample = llh.exp().sum(dim=1).log()
    llh_batch = llh_sample.sum(dim=0)
    llh_batch = llh_batch.cpu().detach().numpy().tolist()

    return llh_batch


def get_log_prob(x, mean, cov):
    """ Get pdf from multivariate gaussian
    :param x: input value
    :param mean: mean of multivariate normal
    :param cov: covariance matrix
    :return:
    """

    dist = MultivariateNormal(loc=mean, covariance_matrix=cov)
    log_prob = dist.log_prob(value=x)

    return log_prob
