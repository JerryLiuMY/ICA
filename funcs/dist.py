from torch.distributions import MultivariateNormal


def get_normal_lp(x, loc, cov_tril):
    """ Get log pdf from multivariate gaussian
    :param x: input value
    :param loc: location of multivariate normal
    :param cov_tril: lower-triangular factor of covariance
    :return:
    """

    # use lower-triangular factor for faster computation
    dist = MultivariateNormal(loc=loc, scale_tril=cov_tril)
    log_prob = dist.log_prob(value=x)

    return log_prob
