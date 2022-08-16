from data_prep.generator import generate_data
from data_prep.loader import load_data
from global_settings import VAE_PATH
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

    # define path and laod parameters
    train_size, valid_size, simu_size = exp_dict["train_size"], exp_dict["valid_size"], exp_dict["simu_size"]
    model_path = os.path.join(VAE_PATH, f"m{m}_n{n}_{''.join([_ for _ in str(activation) if _.isalpha()])}")
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    # training and validation
    train_df = generate_data(m, n, activation, train_size)
    valid_df = generate_data(m, n, activation, valid_size)
    train_loader = load_data(train_df)
    valid_loader = load_data(valid_df)
    model, history = train_vae(m, n, train_loader, valid_loader)
    [train_history, valid_history] = history

    # plot callback and reconstruction
    callback_fig = plot_callback(history)
    simu_df = generate_data(m, n, activation, simu_size)
    simu_loader = load_data(simu_df)
    recon_df = simu_vae(m, n, model, simu_loader)

    # save reconstruction
    torch.save(model.state_dict(), os.path.join(model_path, "model.pth"))
    np.save(os.path.join(model_path, "train_history.npy"), train_history)
    np.save(os.path.join(model_path, "valid_history.npy"), valid_history)
    callback_fig.savefig(os.path.join(model_path, "callback.pdf"), bbox_inches="tight")
    simu_df.to_csv(os.path.join(model_path, "simu_df.csv"))
    recon_df.to_csv(os.path.join(model_path, "recon_df.csv"))


if __name__ == "__main__":
    from torch import nn
    main(m=2, n=20, activation=nn.ReLU())
    main(m=2, n=20, activation=nn.Sigmoid())
    main(m=2, n=20, activation=nn.Tanh())
    main(m=2, n=20, activation=nn.GELU())
    plot_latent_2d(n=20)
