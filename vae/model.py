from params.base import Encoder, Decoder
import torch.nn as nn
import torch


class VAE(nn.Module):
    def __init__(self, m, n):
        super(VAE, self).__init__()
        self.name = "vae"
        self.encoder = Encoder(m, n)
        self.decoder = Decoder(m, n)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        latent = self.latent_sample(mu, logvar)
        mean, logs2 = self.decoder(latent)

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
