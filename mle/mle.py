import torch.nn.functional as F
import torch.nn as nn


class MLEModel(nn.Module):
    def __init__(self, m, n):
        super(MLEModel, self).__init__()
        self.name = "mle"
        self.input_dim = n
        self.latent_dim = m

        # linear layer
        self.inter_dim = self.input_dim
        self.fc = nn.Linear(in_features=self.latent_dim, out_features=self.inter_dim)

        # first mle layer
        self.dec2 = nn.Linear(in_features=self.inter_dim, out_features=self.inter_dim)

        # second mle layer -- reconstruction
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
