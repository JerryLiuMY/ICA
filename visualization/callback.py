from visualization.procrustes import get_procrustes
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import seaborn as sns
import re
import os
sns.set()


def plot_callback(m, n, model_name, exp_path, llh_method):
    """ Plot training and validation history
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param model_name: model name
    :param exp_path: path for experiment
    :param llh_method: method for numerical integration
    :return: dataframe of z and x
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    activations = [nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU()]
    axes = [ax for sub_axes in axes for ax in sub_axes]

    for ax, activation in zip(axes, activations):
        activation_name = ''.join([_ for _ in re.sub("[\(\[].*?[\)\]]", "", str(activation)) if _.isalpha()])
        model_path = os.path.join(exp_path, f"m{m}_n{n}_{activation_name}")
        train_llh = np.load(os.path.join(model_path, f"train_llh_{llh_method}.npy"))
        valid_llh = np.load(os.path.join(model_path, f"valid_llh_{llh_method}.npy"))

        # plot train_llh and valid_llh
        ax.set_title(f"Learning curve of {activation_name} [Integration = {llh_method}]")
        ax.plot(train_llh, color=sns.color_palette()[0], label="train_llh")
        ax.plot(valid_llh, color=sns.color_palette()[1], label="valid_llh")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Log-Likelihood")

        # calculate disparity score
        w_disp, b_disp = get_procrustes(m, n, activation, model_name)
        handle_w_disp = mpatches.Patch(color=sns.color_palette()[7], label=f"w_disp = {round(w_disp, 3)}", alpha=0.8)
        handle_b_disp = mpatches.Patch(color=sns.color_palette()[7], label=f"b_disp = {round(b_disp, 3)}", alpha=0.8)

        ax_ = ax.twinx()
        ax_.grid(False)
        if model_name == "vae":
            train_loss = np.load(os.path.join(model_path, "train_loss.npy"))
            valid_loss = np.load(os.path.join(model_path, "valid_loss.npy"))
            ax_.plot(train_loss, color=sns.color_palette()[2], label="train_loss")
            ax_.plot(valid_loss, color=sns.color_palette()[3], label="valid_loss")
            ax_.set_ylabel("ELBO")

            handles, labels = ax.get_legend_handles_labels()
            handles_, labels_ = ax_.get_legend_handles_labels()
            ax.legend(handles + handles_, labels + labels_, loc="upper right")
            ax_.legend(handles=[handle_w_disp, handle_b_disp], loc="lower right", handlelength=0.2, handletextpad=0.5)
        elif "mle" in model_name:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc="upper right")
            ax_.legend(handles=[handle_w_disp, handle_b_disp], loc="lower right", handlelength=0.2, handletextpad=0.5)
        else:
            raise ValueError("Invalid model name")

    plt.tight_layout()

    return fig
