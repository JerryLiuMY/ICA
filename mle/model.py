from params.base import Decoder
import torch.nn as nn


class MLE(nn.Module):
    def __init__(self, m, n, fit_s2):
        super(MLE, self).__init__()
        self.name, self.m, self.n = "mle", m, n
        self.decoder = Decoder(m, n, fit_s2=fit_s2)

    def forward(self, z):
        mean = self.decoder(z)

        return mean
