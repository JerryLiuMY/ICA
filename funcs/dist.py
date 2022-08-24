from torch.distributions import MultivariateNormal


def get_norm_lp(x, mean, cov):
    """ Get log pdf from multivariate gaussian
    :param x: input value
    :param mean: mean of multivariate normal
    :param cov: covariance matrix
    :return:
    """

    dist = MultivariateNormal(loc=mean, covariance_matrix=cov)
    log_prob = dist.log_prob(value=x)

    return log_prob
