from params.params import mle_dict as train_dict
from datetime import datetime
from global_settings import device
from mle.model import MLE
import numpy as np
import torch


def train_mlesgd(m, n, train_loader, valid_loader, llh_func):
    """ Perform autograd to train the model and find logs2
    :param m: latent dimension
    :param n: observed dimension
    :param train_loader: training dataset loader
    :param valid_loader: validation dataset loader
    :param llh_func: function for numerical integration
    :return: trained model and training loss history
    """

    pass
