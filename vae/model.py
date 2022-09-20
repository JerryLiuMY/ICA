from params.base import Encoder, DecoderDGP, Decoder
from global_settings import DEVICE
from params.params import sigma
import torch.nn as nn
import numpy as np
import torch


class VAE(nn.Module):
    def __init__(self, m, n, activation, fit_s2, decoder_dgp):
        super(VAE, self).__init__()
        self.name, self.m, self.n, self.fit_s2 = "vae", m, n, fit_s2
        self.activation, self.decoder_dgp = activation, decoder_dgp

        self.encoder = Encoder(m, n)
        if self.decoder_dgp:
            self.decoder = DecoderDGP(m, n, fit_s2=self.fit_s2, activation=self.activation)
        else:
            self.decoder = Decoder(m, n, fit_s2=self.fit_s2)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        latent = self.latent_sample(mu, logvar)
        if self.fit_s2:
            x_recon, logs2 = self.decoder(latent)
        else:
            x_recon = self.decoder(latent)

            logs2 = np.repeat(np.log(sigma ** 2), x.shape[0]).reshape(x.shape[0], 1)
            logs2 = torch.tensor(logs2, requires_grad=False).to(DEVICE)

        return x_recon, logs2, mu, logvar

    def latent_sample(self, mu, logvar):
        # the re-parameterization trick
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


def elbo_gaussian(x, x_recon, logs2, mu, logvar, beta):
    """ Calculating loss for variational autoencoder
    :param x: original image
    :param x_recon: reconstruction in the output layer
    :param logs2: log of the variance in the output layer
    :param mu: mean in the fitted variational distribution
    :param logvar: log of the variance in the variational distribution
    :param beta: beta
    :return: reconstruction loss + KL
    """

    # KL-divergence
    kl_div = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # reconstruction loss
    recon_loss = - torch.sum(
        logs2.mul(x.size(dim=1)/2) + torch.norm(x - x_recon, 2, dim=1).pow(2).div(logs2.exp().mul(2))
    )

    # loss
    loss = - beta * kl_div + recon_loss

    return - loss
