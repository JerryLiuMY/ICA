from global_settings import DRIVE_PATH
from global_settings import DATA_PATH
from scipy.spatial import procrustes
from global_settings import PATH_DICT
import pickle5 as pickle
import numpy as np
import torch
import os
import re


def get_procrustes(m, n, activation, model_name):
    """ Get disparity from procrustes analysis
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param activation: activation function for mlp
    :param model_name: name of the model to run
    :return: trained model and training loss history
    """

    # load ground truth w
    with open(os.path.join(DATA_PATH, f"params_{m}_{n}.pkl"), "rb") as handle:
        w = pickle.load(handle)["w"]

    # load weight w_hat
    activation_name = ''.join([_ for _ in re.sub("[\(\[].*?[\)\]]", "", str(activation)) if _.isalpha()])
    model_path = os.path.join(PATH_DICT[model_name], f"m{m}_n{n}_{activation_name}")
    model = torch.load(os.path.join(model_path, "model.pth"))
    w_hat = model["decoder.fc.weight"].numpy().astype(np.float64)
    disparity = procrustes(w, w_hat)[2]

    return disparity
