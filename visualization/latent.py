from global_settings import VAE_PATH
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
sns.set()


def plot_latent_2d(n):
    """ Visualize the reconstructed 2D latent space
    :param n: dimension of the target variable
    :return: dataframe of z and x
    """

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    activations = ["ReLU", "Sigmoid", "Tanh", "GELU"]
    for ax, activation in zip(axes, activations):
        model_path = os.path.join(VAE_PATH, f"m2_n{n}_{activation}")
        recon_df = pd.read_csv(os.path.join(model_path, "recon_df.csv"))
        simu_df = pd.read_csv(os.path.join(model_path, "simu_df.csv"))
        sns.kdeplot(data=simu_df, x="z0", y="z1", fill=True, alpha=1., ax=ax)
        sns.kdeplot(data=recon_df, x="mu0", y="mu1", fill=True, alpha=.7, ax=ax)
        ax_legend_true = mpatches.Patch(color=sns.color_palette()[0], label="Original", alpha=0.8)
        ax_legend_recon = mpatches.Patch(color=sns.color_palette()[1], label="VAE_Recon", alpha=0.8)
        handles = [ax_legend_true, ax_legend_recon]
        ax.legend(handles=handles, loc="upper right", handlelength=0.2, handletextpad=0.5)
        ax.set_title(f"Latent space of {activation}")
        ax.set_xlabel("z0")
        ax.set_ylabel("z1")
    fig.savefig(os.path.join(VAE_PATH, "latent.pdf"), bbox_inches="tight")
