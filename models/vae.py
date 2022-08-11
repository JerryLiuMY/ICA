import torch.nn.functional as F
from params.params import params_dict
import torch.nn as nn
import torch


class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        mu, logvar = self.encoder(x)
        latent = self.latent_sample(mu, logvar)
        x_rec = self.decoder(latent)

        return x_rec, mu, logvar

    def latent_sample(self, mu, logvar):
        # the re-parameterization trick
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.input_size = params_dict["n"]
        self.hidden = params_dict["m"]


class Encoder(Block):
    def __init__(self):
        super(Encoder, self).__init__()

        # first encoder layer
        self.inter_size = self.input_size
        self.enc1 = nn.Linear(in_features=self.inter_size, out_features=self.inter_size // 2)

        # second encoder layer
        self.inter_size = self.inter_size // 2
        self.enc2 = nn.Linear(in_features=self.inter_size, out_features=self.inter_size // 2)

        # map to mu and variance
        self.inter_size = self.inter_size // 2
        self.fc_mu = nn.Linear(in_features=self.inter_size, out_features=self.hidden)
        self.fc_logvar = nn.Linear(in_features=self.inter_size, out_features=self.hidden)

    def forward(self, x):
        # convolution layers
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))

        # calculate mu & logvar
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class Decoder(Encoder, Block):
    def __init__(self):
        super(Decoder, self).__init__()

        # linear layer
        self.fc = nn.Linear(in_features=self.hidden, out_features=self.inter_size)

        # first decoder layer
        self.dec2 = nn.Linear(in_features=self.inter_size, out_features=self.inter_size * 2)
        self.inter_size = self.inter_size * 2

        # second decoder layer -- m
        self.dec1_m = nn.Linear(in_features=self.inter_size, out_features=self.inter_size * 2)
        self.output_size = self.self.inter_size * 2

        # second decoder layer -- s
        self.dec1_s = nn.Linear(in_features=self.inter_size, out_features=1)

    def forward(self, x):
        # linear layer
        x = self.fc(x)

        # convolution layers
        x = F.relu(self.dec2(x))
        m = self.dec1_m(x)
        logs2 = self.dec1_s(x)

        return m, logs2
