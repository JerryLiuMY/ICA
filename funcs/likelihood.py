from scipy.stats import multivariate_normal
from global_settings import device
import numpy as np
import torch


def get_llh_batch(m, input_batch, model):
    """ Find log-likelihood from data and trained model
    :param m: latent dimension
    :param input_batch: inputs related to observation [batch_size x dimension]
    :param model: trained model
    :return: log-likelihood
    """

    # define input
    x_batch, mean_batch, logs2_batch = input_batch
    batch_size = x_batch.shape[0]

    # get numerical integration
    min_lim, max_lim, space = -10, 10, 101
    lin_space = np.linspace(min_lim, max_lim, space)
    grid_space = np.array([0.5 * (lin_space[i] + lin_space[i + 1]) for i in range(len(lin_space) - 1)])
    volume = ((max_lim - min_lim) / (space - 1)) ** m

    llh_batch = 0.
    for batch in range(batch_size):
        x, mean, logs2 = x_batch[batch, :], mean_batch[batch, :], logs2_batch[batch, :]
        s2 = logs2.exp()
        x = x.cpu().detach().numpy()
        s2 = s2.cpu().detach().numpy()

        llh_sample = 0.
        for z_grid in zip(*[list(grid_space) for _ in range(m)]):
            z_grid = torch.tensor(np.array(z_grid).astype(np.float32))
            z_grid = z_grid.to(device)
            x_recon = model.decoder(z_grid)[0]
            z_grid = z_grid.cpu().detach().numpy()
            x_recon = x_recon.cpu().detach().numpy()

            # perform numerical integration
            normpdf = get_normpdf(x, x_recon, s2[0]) * get_normpdf(z_grid, 0, 1)
            llh_sample += normpdf * volume

        llh_batch += llh_sample

    return llh_batch


def get_normpdf(x, mean, s2):
    """ Get pdf from multivariate gaussian
    :param:
    :return:
    """

    normpdf = multivariate_normal.pdf(x=x, mean=mean, cov=s2)

    return normpdf
