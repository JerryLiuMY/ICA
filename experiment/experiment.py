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
from functools import partial
import numpy as np
import torch
import os


def experiment(m, n, activation, model_name, model_path, train_s2, decoder_dgp, llh_method, seed):
    """ Perform experiments for non-linear ICA
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param activation: activation function for mlp
    :param model_name: name of the model to run
    :param model_path: path for model
    :param train_s2: whether to train s2 or not
    :param decoder_dgp: whether to use the same decoder as dgp
    :param llh_method: method for numerical integration
    :param seed: random seed for dgp
    """

    # define training/simulation functions
    train_size, valid_size, simu_size = exp_dict["train_size"], exp_dict["valid_size"], exp_dict["simu_size"]
    train_mleauto = partial(train_mle, grad_method="auto")
    train_mlesgd = partial(train_mle, grad_method="sgd")
    train_dict = {"vae": train_vae, "mleauto": train_mleauto, "mlesgd": train_mlesgd}
    simu_dict = {"vae": simu_vae, "mleauto": simu_mle, "mlesgd": simu_mle}
    llh_dict = {"mc": get_llh_mc, "grid": get_llh_grid, "null": get_llh_null}
    train_func = train_dict[model_name]
    simu_func = simu_dict[model_name]
    llh_func = llh_dict[llh_method]

    # training and validation
    train_df = generate_data(m, n, activation, train_size, seed=seed)
    valid_df = generate_data(m, n, activation, valid_size, seed=seed)
    train_loader = load_data(train_df)
    valid_loader = load_data(valid_df)

    outputs, callback = train_func(m, n, activation, train_loader, valid_loader, train_s2, decoder_dgp, llh_func)
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
    simu_df = generate_data(m, n, activation, simu_size, seed=seed)
    simu_loader = load_data(simu_df)
    recon_df = simu_func(outputs, simu_loader)
    simu_df.to_csv(os.path.join(model_path, "simu_df.csv"))
    recon_df.to_csv(os.path.join(model_path, "recon_df.csv"))
