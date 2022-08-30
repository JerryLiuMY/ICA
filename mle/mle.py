import torch.nn.functional as F
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, m, n):
        super(Block, self).__init__()
        self.input_dim = n
        self.latent_dim = m


class MLEModel(Block):
    def __init__(self, m, n):
        super(MLEModel, self).__init__(m, n)

        # linear layer
        self.fc = nn.Linear(in_features=self.latent_dim, out_features=self.inter_dim)

        # first mle layer
        self.dec2 = nn.Linear(in_features=self.inter_dim, out_features=self.inter_dim)

        # second mle layer -- reconstruction
        self.output_dim = self.inter_dim
        self.dec1 = nn.Linear(in_features=self.inter_dim, out_features=self.output_dim)

    def forward(self, z):
        # linear layer
        inter = self.fc(z)

        # decoder layers
        inter = F.relu(self.dec2(inter))
        recon = self.dec1(inter)

        return recon
