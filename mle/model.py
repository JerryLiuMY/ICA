from params.base import Decoder
import torch.nn as nn


class MLE(nn.Module):
    def __init__(self, m, n):
        super(MLE, self).__init__()
        self.name = "mle"
        self.decoder = Decoder(m, n, fit_s2=False)

    def forward(self, z):
        mean = self.decoder(z)

        return mean
