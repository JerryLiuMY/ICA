from scipy.spatial import procrustes
import numpy as np
import torch
import os
import re


def get_procrustes(m, n, activation, exp_path):
    """ Get disparity from procrustes analysis
      - Rows of the matrix: set of points or vectors
      - Columns of the matrix: dimension of the space
    Given two identically sized matrices, procrustes standardizes both such that:
      - tr(AA^T)=1
      - Both sets of points are centered around the origin
    Procrustes then applies the optimal transform to the second matrix (scaling/dilation, rotations, and reflections)
    to minimize the sum of the squares of the point-wise differences between the two input datasets
                                      M^2=\sum(data_1 - data_2)^2
      - The function was not designed to handle datasets with different number of rows (number of datapoints)
      - The function was able to handle datasets with different number of columns (dimensionality), add columns of zeros
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param activation: activation function for mlp
    :param exp_path: path for experiment
    :return: trained model and training loss history
    """

    # load fitted w_hat and b_hat
    activation_name = ''.join([_ for _ in re.sub("[\(\[].*?[\)\]]", "", str(activation)) if _.isalpha()])
    model_path = os.path.join(exp_path, f"m{m}_n{n}_{activation_name}")
    model = torch.load(os.path.join(model_path, "model.pth"))
    w_hat = model["decoder.fc.weight"].numpy().astype(np.float64)
    b_hat = model["decoder.fc.bias"].numpy().astype(np.float64)

    # perform procrustes analysis
    b = b.reshape(len(b), 1)
    b_hat = b_hat.reshape(len(b_hat), 1)
    w_disp = procrustes(w, w_hat)[2]
    b_disp = procrustes(b, b_hat)[2]

    return w_disp, b_disp
