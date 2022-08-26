from global_settings import VAE_PATH
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import seaborn as sns
import os
sns.set()


def plot_callback(n, llh_method):
    """ Plot training and validation history
    :param n: dimension of the target variable
    :param llh_method: method for numerical integration
    :return: dataframe of z and x
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    activations = [nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.GELU()]
    axes = [ax for sub_axes in axes for ax in sub_axes]

    for ax, activation in zip(axes, activations):
        activation_name = ''.join([_ for _ in str(activation) if _.isalpha()])
        model_path = os.path.join(VAE_PATH, f"m2_n{n}_{activation_name}")
        train_loss = np.load(os.path.join(model_path, "train_loss.npy"))
        valid_loss = np.load(os.path.join(model_path, "valid_loss.npy"))
        train_llh = np.load(os.path.join(model_path, f"train_llh_{llh_method}.npy"))
        valid_llh = np.load(os.path.join(model_path, f"valid_llh_{llh_method}.npy"))

        ax.set_title(f"Learning curve of {activation_name} [Integration = {llh_method}]")
        ax.plot(train_llh, color=sns.color_palette()[0], label="train_llh")
        ax.plot(valid_llh, color=sns.color_palette()[1], label="valid_llh")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Log-Likelihood")

        ax_ = ax.twinx()
        ax_.plot(train_loss, color=sns.color_palette()[2], label="train_loss")
        ax_.plot(valid_loss, color=sns.color_palette()[3], label="valid_loss")
        ax_.set_ylabel("ELBO")
        ax_.grid(False)

        handles, labels = ax.get_legend_handles_labels()
        handles_, labels_ = ax_.get_legend_handles_labels()
        ax.legend(handles + handles_, labels + labels_, loc="upper right")

    plt.tight_layout()

    return fig
