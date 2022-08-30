from params.base import Decoder
import torch.nn as nn


class MLE(nn.Module):
    def __init__(self, m, n):
        super(MLE, self).__init__()
        self.name = "autograd"
        self.decoder = Decoder(m, n)

    def forward(self, z):
        mean, logs2 = self.decoder(z)

        return mean, logs2
