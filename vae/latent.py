from global_settings import VAE_PATH
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from data_prep.posterior import simu_post
from numpy import random
import numpy as np
import pandas as pd
import seaborn as sns
import os
sns.set()
palette = sns.color_palette()


def plot_latent_2d(n):
    """ Visualize the reconstructed 2D latent space
    :param n: dimension of the target variable
    :return: dataframe of z and x
    """

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    activations = ["ReLU", "Sigmoid", "Tanh", "GELU"]
    for ax, activation in zip(axes, activations):
        # build simu_df and recon_df (num. of samples per datapoint set to 1)
        model_path = os.path.join(VAE_PATH, f"m2_n{n}_{activation}")
        simu_df = pd.read_csv(os.path.join(model_path, "simu_df.csv"), index_col=0)
        # simu_df["post0"] = simu_df["logvar0"].apply(lambda _: np.exp(0.5 * _))
        # simu_df["post1"] = simu_df["logvar0"].apply(lambda _: np.exp(0.5 * _))

        recon_df = pd.read_csv(os.path.join(model_path, "recon_df.csv"), index_col=0)
        recon_df["std0"] = recon_df["logvar0"].apply(lambda _: np.exp(0.5 * _))
        recon_df["std1"] = recon_df["logvar1"].apply(lambda _: np.exp(0.5 * _))
        recon_df["post0"] = recon_df.apply(lambda _: random.normal(loc=_.mu0, scale=_.std0, size=1)[0], axis=1)
        recon_df["post1"] = recon_df.apply(lambda _: random.normal(loc=_.mu1, scale=_.std1, size=1)[0], axis=1)

        # visualization of the 2d latent distribution
        sns.kdeplot(data=simu_df, x="z0", y="z1", fill=True, alpha=1., ax=ax)
        sns.kdeplot(data=recon_df, x="post0", y="post1", fill=True, alpha=.7, ax=ax)
        ax_legend_prior = mpatches.Patch(color=palette[0], label="Prior $p(z)$", alpha=0.8)
        ax_legend_recon = mpatches.Patch(color=palette[1], label="Recon $\widehat{p}(z|x)$", alpha=0.8)
        handles = [ax_legend_prior, ax_legend_recon]
        ax.legend(handles=handles, loc="upper right", handlelength=0.2, handletextpad=0.5)
        ax.set_title(f"Latent space of {activation}")
        ax.set_xlabel("z0")
        ax.set_ylabel("z1")

    plt.tight_layout()

    return fig
