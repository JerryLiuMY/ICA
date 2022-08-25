from global_settings import VAE_PATH
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
sns.set()


def plot_callback(n):
    """ Plot training and validation history
    :param n: dimension of the target variable
    :return: dataframe of z and x
    """

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    activations = ["ReLU", "Sigmoid", "Tanh", "GELU"]
    axes = [ax for sub_axes in axes for ax in sub_axes]

    for ax, activation in zip(axes, activations):
        model_path = os.path.join(VAE_PATH, f"m2_n{n}_{activation}")
        train_loss = np.load(os.path.join(model_path, "train_loss.npy"))
        valid_loss = np.load(os.path.join(model_path, "valid_loss.npy"))
        train_llh = np.load(os.path.join(model_path, "train_llh.npy"))
        valid_llh = np.load(os.path.join(model_path, "valid_llh.npy"))

        ax.set_title(f"Learning curve of {activation}")
        ax.plot(train_llh, color=sns.color_palette()[0], label="train_llh")
        ax.plot(valid_llh, color=sns.color_palette()[1], label="valid_llh")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Log-Likelihood")

        ax_ = ax.twinx()
        ax_.plot(train_loss, color=sns.color_palette()[2], label="train_loss")
        ax_.plot(valid_loss, color=sns.color_palette()[3], label="valid_loss")
        ax_.set_ylabel("ELBO")
        ax_.grid(False)

        h, l = ax.get_legend_handles_labels()
        h_, l_ = ax_.get_legend_handles_labels()
        ax.legend(h + h_, l + l_, loc="upper right")

    plt.tight_layout()

    return fig
