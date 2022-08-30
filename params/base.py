from torch import nn as nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, m, n):
        super(Block, self).__init__()
        self.input_dim = n
        self.latent_dim = m
        self.output_dim = n


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


class Decoder(Block):
    def __init__(self, m, n):
        super(Decoder, self).__init__(m, n)

        # linear layer
        self.inter_dim = self.output_dim
        self.fc = nn.Linear(in_features=self.latent_dim, out_features=self.inter_dim)

        # first decoder layer
        self.dec2 = nn.Linear(in_features=self.inter_dim, out_features=self.inter_dim)

        # second decoder layer -- mean and logs2
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
