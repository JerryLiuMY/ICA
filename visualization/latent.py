from global_settings import VAE_PATH
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

    # initialize figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    activations = ["ReLU", "Sigmoid", "Tanh", "GELU"]
    for ax, activation in zip(axes, activations):
        model_path = os.path.join(VAE_PATH, f"m2_n{n}_{activation}")
        recon_df = pd.read_csv(os.path.join(model_path, "recon_df.csv"))
        simu_df = pd.read_csv(os.path.join(model_path, "simu_df.csv"))
        sns.kdeplot(data=simu_df, x="z0", y="z1", fill=True, alpha=.5, label="true", ax=ax)
        sns.kdeplot(data=recon_df, x="mu0", y="mu1", fill=True, alpha=.5, label="recon", ax=ax)
        ax.set_xlabel("z0")
        ax.set_ylabel("z1")
        ax.legend(loc="upper right")
