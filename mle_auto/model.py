from params.base import Decoder
import torch.nn as nn


class MLEAuto(nn.Module):
    def __init__(self, m, n):
        super(MLEAuto, self).__init__()
        self.name = "autograd"
        self.decoder = Decoder(m, n)

    def forward(self, z):
        mean, logs2 = self.decoder(z)

        return mean, logs2
