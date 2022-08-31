from global_settings import path_dict
from likelihood.llh import get_llh_mc, get_llh_grid
from data_prep.generator import generate_data
from data_prep.loader import load_data
from vae.training import train_vae
from mle.training_auto import train_mleauto
from mle.training_sgd import train_mlesgd
from vae.simulation import simu_vae
from mle.simulation import simu_mle
from params.params import exp_dict
from visualization.callback import plot_callback
from visualization.latent import plot_latent_2d
from visualization.recon import plot_recon_2d
import numpy as np
import torch
import os


def main(m, n, activation, model_name, llh_method):
    """ Perform experiments for non-linear ICA
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param activation: activation function for mlp
    :param model_name: name of the model to run
    :param llh_method: method for numerical integration
    """

    # define path and load parameters
    train_size, valid_size, simu_size = exp_dict["train_size"], exp_dict["valid_size"], exp_dict["simu_size"]
    activation_name = ''.join([_ for _ in str(activation) if _.isalpha()])
    model_path = os.path.join(path_dict[model_name], f"m{m}_n{n}_{activation_name}")
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    # define training/simulation functions
    train_dict = {"vae": train_vae, "mleauto": train_mleauto, "mlesgd": train_mlesgd}
    simu_dict = {"vae": simu_vae, "mleauto": simu_mle, "mlesgd": simu_mle}
    llh_dict = {"mc": get_llh_mc, "grid": get_llh_grid}
    train_func = train_dict[model_name]
    simu_func = simu_dict[model_name]
    llh_func = llh_dict[llh_method]

    # training and validation
    train_df = generate_data(m, n, activation, train_size)
    valid_df = generate_data(m, n, activation, valid_size)
    train_loader = load_data(train_df)
    valid_loader = load_data(valid_df)

    model, callback = train_func(m, n, train_loader, valid_loader, llh_func)
    torch.save(model.state_dict(), os.path.join(model_path, "model.pth"))
    if "loss" in callback.keys():
        [train_loss, valid_loss] = callback["loss"]
        np.save(os.path.join(model_path, "train_loss.npy"), train_loss)
        np.save(os.path.join(model_path, "valid_loss.npy"), valid_loss)
    if "llh" in callback.keys():
        [train_llh, valid_llh] = callback["llh"]
        np.save(os.path.join(model_path, f"train_llh_{llh_method}.npy"), train_llh)
        np.save(os.path.join(model_path, f"valid_llh_{llh_method}.npy"), valid_llh)

    # run simulation and reconstruction
    simu_df = generate_data(m, n, activation, simu_size)
    simu_loader = load_data(simu_df)
    recon_df = simu_func(m, n, model, simu_loader)
    simu_df.to_csv(os.path.join(model_path, "simu_df.csv"))
    recon_df.to_csv(os.path.join(model_path, "recon_df.csv"))


def plotting(m, n, model_name, llh_method):
    """ Plot original space, latent space and callback
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param model_name: name of the model to run
    :param llh_method: method for numerical integration
    """

    # define path and load parameters
    figure_path = os.path.join(path_dict[model_name], f"m{m}_n{n}_figure")
    if not os.path.isdir(figure_path):
        os.mkdir(figure_path)

    # plot recon, latent and callback
    recon = plot_recon_2d(m, n)
    latent = plot_latent_2d(m, n)
    callback = plot_callback(m, n, llh_method=llh_method)
    recon.savefig(os.path.join(figure_path, f"recon_m{m}_n{n}.pdf"), bbox_inches="tight")
    latent.savefig(os.path.join(figure_path, f"latent_m{m}_n{n}.pdf"), bbox_inches="tight")
    callback.savefig(os.path.join(figure_path, f"callback_m{m}_n{n}_{llh_method}.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    from torch import nn
    main(m=2, n=10, activation=nn.ReLU(), model_name="mleauto", llh_method="mc")
    main(m=2, n=10, activation=nn.Sigmoid(), model_name="mleauto", llh_method="mc")
    main(m=2, n=10, activation=nn.Tanh(), model_name="mleauto", llh_method="mc")
    main(m=2, n=10, activation=nn.GELU(), model_name="mleauto", llh_method="mc")
    plotting(m=2, n=10, model_name="mleauto", llh_method="mc")
