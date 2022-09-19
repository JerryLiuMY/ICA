import torch


def get_llh_null(m, n, x, model, logs2):
    """ Find log-likelihood from data and trained model [mc]
    :param m: latent dimension
    :param n: observed dimension
    :param x: inputs related to the observation x data
    :param model: trained model
    :param logs2: log of the estimated s2
    :return: log-likelihood
    """

    llh_sample = torch.zeros(x.shape[0])

    print(x.shape)
    print(llh_sample.shape)

    return llh_sample
