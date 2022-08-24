from global_settings import VAE_PATH
from data_prep.generator import generate_data
from data_prep.loader import load_data
from vae.training import train_vae
from visualization.callback import plot_callback
from vae.simulation import simu_vae
from params.params import exp_dict
from visualization.latent import plot_latent_2d
import torch
import numpy as np
import os


def main(m, n, activation):
    """ Perform experiments with for non-linear ICA
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param activation: activation function for mlp
    :return: dataframe of z and x
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
    [train_loss, valid_loss] = loss
    [train_llh, valid_llh] = llh

    # plot callback and reconstruction
    callback = plot_callback(loss, llh)
    simu_df = generate_data(m, n, activation, simu_size)
    simu_loader = load_data(simu_df)
    recon_df = simu_vae(m, n, model, simu_loader)

    # save reconstruction
    torch.save(model.state_dict(), os.path.join(model_path, "model.pth"))
    np.save(os.path.join(model_path, "train_loss.npy"), train_loss)
    np.save(os.path.join(model_path, "valid_loss.npy"), valid_loss)
    np.save(os.path.join(model_path, "train_llh.npy"), train_llh)
    np.save(os.path.join(model_path, "valid_llh.npy"), valid_llh)
    callback.savefig(os.path.join(model_path, "callback.pdf"), bbox_inches="tight")
    simu_df.to_csv(os.path.join(model_path, "simu_df.csv"))
    recon_df.to_csv(os.path.join(model_path, "recon_df.csv"))


if __name__ == "__main__":
    from torch import nn
    main(m=2, n=20, activation=nn.ReLU())
    main(m=2, n=20, activation=nn.Sigmoid())
    main(m=2, n=20, activation=nn.Tanh())
    main(m=2, n=20, activation=nn.GELU())
    plot_latent_2d(n=20)
