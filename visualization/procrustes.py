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

    # load ground truth w and b
    with open(os.path.join(DATA_PATH, f"params_{m}_{n}.pkl"), "rb") as handle:
        params = pickle.load(handle)
        w, b = params["w"], params["b"]

    # load weight w_hat and b_hat
    activation_name = ''.join([_ for _ in re.sub("[\(\[].*?[\)\]]", "", str(activation)) if _.isalpha()])
    model_path = os.path.join(PATH_DICT[model_name], f"m{m}_n{n}_{activation_name}")
    model = torch.load(os.path.join(model_path, "model.pth"))
    w_hat = model["decoder.fc.weight"].numpy().astype(np.float64)
    b_hat = model["decoder.fc.bias"].numpy().astype(np.float64)

    # perform procrustes analysis
    b = b.reshape(len(b), 1)
    b_hat = b_hat.reshape(len(b_hat), 1)
    w_disparity = procrustes(w, w_hat)[2]
    b_disparity = procrustes(b, b_hat)[2]

    return w_disparity, b_disparity
