from sklearn.cross_decomposition import CCA
from tools.utils import activation2name
from scipy.spatial import procrustes
import pandas as pd
import scipy.stats
import os


def get_metrics(m, n, activation, exp_path):
    """ Get disparity from procrustes analysis and correlation from canonical correlation analysis
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param activation: activation function for mlp
    :param exp_path: path for experiment
    :return: disparity score and correlation
    """

    # load x and mean
    activation_name = activation2name(activation)
    model_path = os.path.join(exp_path, f"m{m}_n{n}_{activation_name}")
    simu_df = pd.read_csv(os.path.join(model_path, "simu_df.csv"))
    recon_df = pd.read_csv(os.path.join(model_path, "recon_df.csv"))
    disp, corr = compute_metrics(simu_df, recon_df)

    return disp, corr


def compute_metrics(simu_df, recon_df):
    """ Compute disparity from procrustes analysis and correlation from canonical correlation analysis
    :param simu_df: simulation df for the original data
    :param recon_df: reconstruction df for the reconstructed data
    :return: disparity score and correlation
    """

    """ Procruste's analysis
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
    """

    # load x and mean
    x = simu_df.loc[:, [_ for _ in simu_df.columns if "x" in _]]
    mean = recon_df.loc[:, [_ for _ in recon_df.columns if "mean" in _]]

    # perform procrustes analysis and cca
    disp = procrustes(x, mean)[2]
    cca = CCA(n_components=1)
    x_c, mean_c = cca.fit_transform(x, mean)
    corr, _ = scipy.stats.pearsonr(x_c.reshape(-1), mean_c.reshape(-1))

    return disp, corr
