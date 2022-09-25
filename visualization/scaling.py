from tools.params import get_li_m_n
from tools.utils import activation2name
from visualization.metrics import compute_metrics
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import seaborn as sns
import itertools
import pandas as pd
import os
sns.set()


def plot_scaling(exp_path, dist, scale):
    """ Plot the scaling of the disparity score & correlation with m and
    :param exp_path: path of experiments
    :param dist: distribution function for generating z
    :param scale: scale of the distribution for generating z
    :return:
    """

    # define m_n_iter & m_iter_n
    m_n_li, m_li_n = get_li_m_n("vae")
    m_n_iter = sorted(list(itertools.product(*m_n_li)))
    m_iter_n = sorted(list(itertools.product(*m_li_n)))

    # generate figures
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Scaling of metrics with simulation data -- dist = {dist} and scale = {scale}")

    disp_dict, corr_dict = build_metric_dicts(m_n_iter, exp_path, dist, scale)
    title = f"n [Fixing m={[_[0] for _ in m_n_iter][0]}]"
    xlabel, xticklabels = "Observed dimension n", [_[1] for _ in m_n_iter]
    plot_scaling_ax(disp_dict, ax=axes[0, 0], ax_info=[title, xlabel, xticklabels, "Procrustes disparity"])
    plot_scaling_ax(corr_dict, ax=axes[0, 1], ax_info=[title, xlabel, xticklabels, "CCA correlation"])

    disp_dict, corr_dict = build_metric_dicts(m_iter_n, exp_path, dist, scale)
    title = f"m [Fixing n={[_[1] for _ in m_iter_n][0]}]"
    xlabel, xticklabels = "Latent dimension m", [_[0] for _ in m_iter_n]
    plot_scaling_ax(disp_dict, ax=axes[1, 0], ax_info=[title, xlabel, xticklabels, "Procrustes disparity"])
    plot_scaling_ax(corr_dict, ax=axes[1, 1], ax_info=[title, xlabel, xticklabels, "CCA correlation"])

    plt.tight_layout()

    return fig


def plot_scaling_ax(metric_dict, ax, ax_info):
    """ Plot each individual image for plot_scaling
    :param metric_dict: dictionary of disparity
    :param ax: axis for plotting
    :param ax_info: information for axis
    :return:
    """

    # define parameters
    title, xlabel, xticklabels, metric = ax_info
    activation_li = [nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU()]
    activation_name_li = [activation2name(activation) for activation in activation_li]

    # make plot for ax
    ax.set_title(f"Scaling of {metric} with {title}")
    for activation_name in activation_name_li:
        ax.plot(metric_dict[activation_name], label=activation_name)

    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric)
    ax.legend(loc="lower right")


def build_metric_dicts(iterable, exp_path, dist, scale):
    """ Plotting function for each individual axis in plot_scaling
    :param iterable: iterator of m, n pairs
    :param exp_path: path of experiments
    :param dist: distribution function for generating z
    :param scale: scale of the distribution for generating z
    :return:
    """

    # define activations
    activation_li = [nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU()]
    activation_name_li = [activation2name(activation) for activation in activation_li]

    # build dictionary of disp & corr
    disp_dict = {activation_name: [] for activation_name in activation_name_li}
    corr_dict = {activation_name: [] for activation_name in activation_name_li}

    for m, n in iterable:
        for activation_name in activation_name_li:
            model_path = os.path.join(exp_path, f"m{m}_n{n}_{activation_name}")
            simu_df = pd.read_csv(os.path.join(model_path, f"simu_df({dist})[{scale}].csv"))
            recon_df = pd.read_csv(os.path.join(model_path, f"recon_df({dist})[{scale}].csv"))
            disp, corr = compute_metrics(simu_df, recon_df)
            disp_dict[activation_name].append(disp)
            corr_dict[activation_name].append(corr)

    return disp_dict, corr_dict
