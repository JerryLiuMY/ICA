import torch.nn.functional as F
import torch.nn as nn
import torch


class VariationalAutoencoder(nn.Module):
    def __init__(self, m, n):
        super(VariationalAutoencoder, self).__init__()
        self.name = "vae"
        self.encoder = Encoder(m, n)
        self.decoder = Decoder(m, n)
        if not self.encoder.input_dim == self.decoder.output_dim:
            raise ValueError("Input size does not match with output size")

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


class Block(nn.Module):
    def __init__(self, m, n):
        super(Block, self).__init__()
        self.input_dim = n
        self.latent_dim = m


class Encoder(Block):
    def __init__(self, m, n):
        super(Encoder, self).__init__(m, n)

        # first encoder layer
        self.inter_dim = self.input_dim
        self.enc1 = nn.Linear(in_features=self.inter_dim, out_features=self.inter_dim)

        # second encoder layer
        self.enc2 = nn.Linear(in_features=self.inter_dim, out_features=self.inter_dim)

        # map to mu and variance
        self.fc_mu = nn.Linear(in_features=self.inter_dim, out_features=self.latent_dim)
        self.fc_logvar = nn.Linear(in_features=self.inter_dim, out_features=self.latent_dim)

    def forward(self, x):
        # encoder layers
        inter = F.relu(self.enc1(x))
        inter = F.relu(self.enc2(inter))

        # calculate mu & logvar
        mu = self.fc_mu(inter)
        logvar = self.fc_logvar(inter)

        return mu, logvar


class Decoder(Encoder, Block):
    def __init__(self, m, n):
        super(Decoder, self).__init__(m, n)

        # linear layer
        self.fc = nn.Linear(in_features=self.latent_dim, out_features=self.inter_dim)

        # first decoder layer
        self.dec2 = nn.Linear(in_features=self.inter_dim, out_features=self.inter_dim)

        # second decoder layer -- mean and logs2
        self.output_dim = self.inter_dim
        self.dec1_mean = nn.Linear(in_features=self.inter_dim, out_features=self.output_dim)
        self.dec1_logs2 = nn.Linear(in_features=self.inter_dim, out_features=1)

    def forward(self, z):
        # linear layer
        inter = self.fc(z)

        # decoder layers
        inter = F.relu(self.dec2(inter))
        mean = self.dec1_mean(inter)
        logs2 = self.dec1_logs2(inter)

        return mean, logs2


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
