from params.base import DecoderDGP, Decoder
import torch.nn as nn


class MLE(nn.Module):
    def __init__(self, m, n, activation, fit_s2, decoder_dgp):
        super(MLE, self).__init__()
        self.name, self.m, self.n, self.fit_s2 = "mle", m, n, fit_s2
        self.activation, self.decoder_dgp = activation, decoder_dgp

        if self.decoder_dgp:
            self.decoder = DecoderDGP(m, n, fit_s2=self.fit_s2, activation=self.activation)
        else:
            self.decoder = Decoder(m, n, fit_s2=self.fit_s2)

    def forward(self, z):
        mean = self.decoder(z)

        return mean
