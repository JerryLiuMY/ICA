from global_settings import VAE_PATH
import matplotlib.patches as mpatches
from data_prep.posterior import simu_post
import matplotlib.pyplot as plt
from torch import nn
from numpy import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import os
sns.set()
palette = sns.color_palette()
tqdm.pandas()


def plot_latent_2d(n):
    """ Visualize the reconstructed 2D latent space
    :param n: dimension of the target variable
    :return: dataframe of z and x
    """

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    activations = [nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.GELU()]
    for ax, activation in zip(axes, activations):
        # build simu_df (num. of samples per datapoint set to 1)
        activation_name = ''.join([_ for _ in str(activation) if _.isalpha()])
        model_path = os.path.join(VAE_PATH, f"m2_n{n}_{activation_name}")
        simu_df = pd.read_csv(os.path.join(model_path, "simu_df.csv"), index_col=0)
        temp_df = simu_df[[col for col in simu_df.columns if "x" in col]]
        post_df = temp_df.progress_apply(lambda _: simu_post(_.values, 2, n, activation), axis=1)
        simu_df[["post0", "post1"]] = pd.DataFrame.from_dict(dict(zip(post_df.index, post_df.values))).T

        # build recon_df (num. of samples per datapoint set to 1)
        recon_df = pd.read_csv(os.path.join(model_path, "recon_df.csv"), index_col=0)
        recon_df["std0"] = recon_df["logvar0"].apply(lambda _: np.exp(0.5 * _))
        recon_df["std1"] = recon_df["logvar1"].apply(lambda _: np.exp(0.5 * _))
        recon_df["post0"] = recon_df.apply(lambda _: random.normal(loc=_.mu0, scale=_.std0, size=1)[0], axis=1)
        recon_df["post1"] = recon_df.apply(lambda _: random.normal(loc=_.mu1, scale=_.std1, size=1)[0], axis=1)

        # visualization of the 2d latent distribution
        sns.kdeplot(data=simu_df, x="z0", y="z1", color=palette[0], fill=True, alpha=1., ax=ax)
        sns.kdeplot(data=simu_df, x="post0", y="post1", color=palette[2], fill=True, alpha=.85, ax=ax)
        sns.kdeplot(data=recon_df, x="post0", y="post1", color=palette[1], fill=True, alpha=.7, ax=ax)
        ax_legend_prior = mpatches.Patch(color=palette[0], label="Prior $p(z)$", alpha=0.8)
        ax_legend_post = mpatches.Patch(color=palette[2], label="Posterior $p(z|x)$", alpha=0.8)
        ax_legend_recon = mpatches.Patch(color=palette[1], label="Recon $\widehat{p}(z|x)$", alpha=0.8)
        handles = [ax_legend_prior, ax_legend_post, ax_legend_recon]
        ax.legend(handles=handles, loc="upper right", handlelength=0.2, handletextpad=0.5)
        ax.set_title(f"Latent space of {activation_name}")
        ax.set_xlabel("z0")
        ax.set_ylabel("z1")

    plt.tight_layout()

    return fig
