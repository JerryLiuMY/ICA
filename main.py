from global_settings import VAE_PATH
from data_prep.generator import generate_data
from data_prep.loader import load_data
from vae.training import train_vae
from visualization.callback import plot_callback
from vae.simulation import simu_vae
from params.params import exp_dict
from visualization.latent import plot_latent_2d
from visualization.recon import plot_recon_2d
import torch
import numpy as np
import os


def main_vae(m, n, activation):
    """ Perform experiments for non-linear ICA
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param activation: activation function for mlp
    """

    # define path and load parameters
    train_size, valid_size, simu_size = exp_dict["train_size"], exp_dict["valid_size"], exp_dict["simu_size"]
    model_path = os.path.join(VAE_PATH, f"m{m}_n{n}_{''.join([_ for _ in str(activation) if _.isalpha()])}")
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    # training and validation
    train_df = generate_data(m, n, activation, train_size)
    valid_df = generate_data(m, n, activation, valid_size)
    train_loader = load_data(train_df)
    valid_loader = load_data(valid_df)
    model, loss, llh = train_vae(m, n, train_loader, valid_loader)
    [train_loss, valid_loss], [train_llh, valid_llh] = loss, llh

    torch.save(model.state_dict(), os.path.join(model_path, "model.pth"))
    np.save(os.path.join(model_path, "train_loss.npy"), train_loss)
    np.save(os.path.join(model_path, "valid_loss.npy"), valid_loss)
    np.save(os.path.join(model_path, "train_llh.npy"), train_llh)
    np.save(os.path.join(model_path, "valid_llh.npy"), valid_llh)

    # run simulation and reconstruction
    simu_df = generate_data(m, n, activation, simu_size)
    simu_loader = load_data(simu_df)
    recon_df = simu_vae(m, n, model, simu_loader)
    simu_df.to_csv(os.path.join(model_path, "simu_df.csv"))
    recon_df.to_csv(os.path.join(model_path, "recon_df.csv"))


def plot(m, n):
    """ Plot original space, latent space and callback
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    """

    # define path and load parameters
    figure_path = os.path.join(VAE_PATH, f"m{m}_n{n}_figure")
    if not os.path.isdir(figure_path):
        os.mkdir(figure_path)

    # plot recon, latent and callback
    recon = plot_recon_2d(n)
    latent = plot_latent_2d(n)
    callback = plot_callback(n)
    recon.savefig(os.path.join(figure_path, f"recon_m2_n{n}.pdf"), bbox_inches="tight")
    latent.savefig(os.path.join(figure_path, f"latent_m2_n{n}.pdf"), bbox_inches="tight")
    callback.savefig(os.path.join(figure_path, f"callback_m2_n{n}.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    from torch import nn
    # main_vae(m=2, n=19, activation=nn.ReLU())
    # main_vae(m=2, n=19, activation=nn.Sigmoid())
    # main_vae(m=2, n=19, activation=nn.Tanh())
    # main_vae(m=2, n=19, activation=nn.GELU())
    plot(m=2, n=2)
