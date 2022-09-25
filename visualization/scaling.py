from tools.params import get_li_m_n
from tools.utils import activation2name
from experiment.experiment import simulation
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import seaborn as sns
import itertools
sns.set()


def plot_scaling(exp_path):
    """ Plot the scaling of the disparity score & correlation with m and
    :param exp_path: path of experiments
    :return:
    """

    # define m_n_iter & m_iter_n
    m_n_li, m_li_n = get_li_m_n("vae")
    m_n_iter = sorted(list(itertools.product(*m_n_li)))
    m_iter_n = sorted(list(itertools.product(*m_li_n)))

    # generate figures
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    disp_dict, corr_dict = build_metric_dicts(m_n_iter, exp_path)
    title = f"n [Fixing m={[_[0] for _ in m_n_iter][0]}]"
    xlabel, xticklabels = "Observed dimension n", [_[1] for _ in m_n_iter]
    plot_scaling_ax(disp_dict, ax=axes[0, 0], ax_info=[title, xlabel, xticklabels, "Procrustes disparity"])
    plot_scaling_ax(corr_dict, ax=axes[0, 1], ax_info=[title, xlabel, xticklabels, "CCA correlation"])

    disp_dict, corr_dict = build_metric_dicts(m_iter_n, exp_path)
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


def build_metric_dicts(iterable, exp_path):
    """ Plotting function for each individual axis in plot_scaling
    :param iterable: iterator of m, n pairs
    :param exp_path: path of experiments
    :return:
    """

    # define activations
    activation_li = [nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU()]
    activation_name_li = [activation2name(activation) for activation in activation_li]

    # build dictionary of disp & corr
    disp_dict = {activation_name: [] for activation_name in activation_name_li}
    corr_dict = {activation_name: [] for activation_name in activation_name_li}

    for m, n in iterable:
        for activation in activation_li:
            simulation(m, n, activation, model_name, outputs, seed, dist, scale)
            disp_dict[activation_name].append(metrics[activation_name][0])
            corr_dict[activation_name].append(metrics[activation_name][1])

    return disp_dict, corr_dict
