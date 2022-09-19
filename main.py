from global_settings import PATH_DICT
from likelihoods.llh_mc import get_llh_mc
from likelihoods.llh_grid import get_llh_grid
from likelihoods.llh_null import get_llh_null
from data_prep.generator import generate_data
from data_prep.loader import load_data
from vae.training import train_vae
from mle.training import train_mle
from vae.simulation import simu_vae
from mle.simulation import simu_mle
from params.params import exp_dict
from visualization.callback import plot_callback
from visualization.recon import plot_recon_2d
from functools import partial
import numpy as np
import torch
import re
import os


def main(m, n, activation, model_name, train_s2, decoder_dgp, llh_method, exp_mode=False):
    """ Perform experiments for non-linear ICA
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param activation: activation function for mlp
    :param model_name: name of the model to run
    :param train_s2: whether to train s2 or not
    :param decoder_dgp: whether to use the same decoder as dgp
    :param llh_method: method for numerical integration
    :param exp_mode: whether in experiment model
    """

    # define path and load parameters
    train_size, valid_size, simu_size = exp_dict["train_size"], exp_dict["valid_size"], exp_dict["simu_size"]
    activation_name = ''.join([_ for _ in re.sub("[\(\[].*?[\)\]]", "", str(activation)) if _.isalpha()])
    model_path = os.path.join(PATH_DICT[model_name], f"m{m}_n{n}_{activation_name}")
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    # define training/simulation functions
    train_mleauto = partial(train_mle, grad_method="auto")
    train_mlesgd = partial(train_mle, grad_method="sgd")
    train_dict = {"vae": train_vae, "mleauto": train_mleauto, "mlesgd": train_mlesgd}
    simu_dict = {"vae": simu_vae, "mleauto": simu_mle, "mlesgd": simu_mle}
    llh_dict = {"mc": get_llh_mc, "grid": get_llh_grid, "null": get_llh_null}
    train_func = train_dict[model_name]
    simu_func = simu_dict[model_name]
    llh_func = llh_dict["null"] if exp_mode and model_name == "vae" else llh_dict[llh_method]

    # training and validation
    train_df = generate_data(m, n, activation, train_size)
    valid_df = generate_data(m, n, activation, valid_size)
    train_loader = load_data(train_df)
    valid_loader = load_data(valid_df)
    decoder_info = [decoder_dgp, activation_name]

    outputs, callback = train_func(m, n, train_loader, valid_loader, train_s2, decoder_info, llh_func)
    model = outputs[0]
    torch.save(model.state_dict(), os.path.join(model_path, "model.pth"))
    if len == 2:
        logs2 = outputs[1].cpu().detach()
        torch.save(logs2, os.path.join(model_path, "logs2.pt"))
    if "llh" in callback.keys():
        [train_llh, valid_llh] = callback["llh"]
        np.save(os.path.join(model_path, f"train_llh_{llh_method}.npy"), train_llh)
        np.save(os.path.join(model_path, f"valid_llh_{llh_method}.npy"), valid_llh)
    if "loss" in callback.keys():
        [train_loss, valid_loss] = callback["loss"]
        np.save(os.path.join(model_path, "train_loss.npy"), train_loss)
        np.save(os.path.join(model_path, "valid_loss.npy"), valid_loss)

    # run simulation and reconstruction
    simu_df = generate_data(m, n, activation, simu_size)
    simu_loader = load_data(simu_df)
    recon_df = simu_func(outputs, simu_loader)
    simu_df.to_csv(os.path.join(model_path, "simu_df.csv"))
    recon_df.to_csv(os.path.join(model_path, "recon_df.csv"))


def plotting(m, n, model_name, llh_method):
    """ Plot original space, latent space and callback
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param model_name: model name
    :param llh_method: method for numerical integration
    """

    # define path and load parameters
    figure_path = os.path.join(PATH_DICT[model_name], f"m{m}_n{n}_figure")
    if not os.path.isdir(figure_path):
        os.mkdir(figure_path)

    # plot recon, latent and callback
    callback = plot_callback(m, n, model_name, llh_method=llh_method)
    callback.savefig(os.path.join(figure_path, f"callback_m{m}_n{n}_{llh_method}.pdf"), bbox_inches="tight")
    recon = plot_recon_2d(m, n, model_name)
    recon.savefig(os.path.join(figure_path, f"recon_m{m}_n{n}.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    from torch import nn
    main(m=2, n=2, activation=nn.ReLU(), model_name="mleauto", train_s2=False, decoder_dgp=True, llh_method="mc")
    # main(m=2, n=10, activation=nn.Sigmoid(), model_name="vae", train_s2=False, decoder_dgp=True, llh_method="mc")
    # main(m=2, n=10, activation=nn.Tanh(), model_name="vae", train_s2=False, decoder_dgp=True, llh_method="mc")
    # main(m=2, n=10, activation=nn.LeakyReLU(), model_name="vae", train_s2=False, decoder_dgp=True, llh_method="mc")
    # plotting(m=2, n=10, model_name="mleauto", llh_method="mc")
