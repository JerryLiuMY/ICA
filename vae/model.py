from params.base import Encoder, DecoderDGP, Decoder
from global_settings import DATA_PATH
from global_settings import device
import pickle5 as pickle
import torch.nn as nn
import numpy as np
import torch
import os


class VAE(nn.Module):
    def __init__(self, m, n, decoder_dgp, fit_s2):
        super(VAE, self).__init__()
        self.name, self.m, self.n = "vae", m, n
        self.match_decoder = decoder_dgp
        self.fit_s2 = fit_s2
        self.encoder = Encoder(m, n)
        if decoder_dgp:
            self.decoder = DecoderDGP(m, n, fit_s2=self.fit_s2)
        else:
            self.decoder = Decoder(m, n, fit_s2=self.fit_s2)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        latent = self.latent_sample(mu, logvar)
        if self.fit_s2:
            mean, logs2 = self.decoder(latent)
        else:
            mean = self.decoder(latent)
            params_path = os.path.join(DATA_PATH, f"params_{self.m}_{self.n}.pkl")
            with open(params_path, "rb") as handle:
                params = pickle.load(handle)
                sigma = params["sigma"]
            logs2 = np.repeat(np.log(sigma ** 2), x.shape[0]).reshape(x.shape[0], 1)
            logs2 = torch.tensor(logs2, requires_grad=False).to(device)

        return mean, logs2, mu, logvar

    def latent_sample(self, mu, logvar):
        # the re-parameterization trick
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


def elbo_gaussian(x, mean, logs2, mu, logvar, beta):
    """ Calculating loss for variational autoencoder
    :param x: original image
    :param mean: mean in the output layer
    :param logs2: log of the variance in the output layer
    :param mu: mean in the hidden layer
    :param logvar: log of the variance in the hidden layer
    :param beta: beta
    :return: reconstruction loss + KL
    """

    # KL-divergence
    kl_div = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # reconstruction loss
    recon_loss = - torch.sum(logs2.mul(x.size(dim=1)/2) + torch.norm(x - mean, 2, dim=1).pow(2).div(logs2.exp().mul(2)))

    # loss
    loss = - beta * kl_div + recon_loss

    return - loss
