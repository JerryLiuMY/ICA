from data.generator import generate_data
from data.loader import load_data
from global_settings import OUTPUT_PATH
from models.train import train_vae
from models.train import valid_vae
import torch
import numpy as np
import os


def experiment(m, n, activation, train_size, valid_size):
    """ Perform experiments with for non-linear ICA
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param activation: activation function for mlp
    :param train_size: number of samples in the training set
    :param valid_size: number of samples in the validation set
    :return: dataframe of z and x
    """

    train_df, valid_df = generate_data(m, n, activation, train_size, valid_size)
    train_loader, valid_loader = load_data(train_df, valid_df)
    model, train_loss = train_vae(train_loader)
    valid_loss = valid_vae(model, valid_loader)

    model_path = os.path.join(OUTPUT_PATH, f"m{m}_n{n}_{''.join([_ for _ in str(activation) if _.isalpha()])}")
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    torch.save(model, os.path.join(model_path, "model.pth"))
    np.save(os.path.join(model_path, "train_loss.npy"), train_loss)
    np.save(os.path.join(model_path, "valid_loss.npy"), valid_loss)
