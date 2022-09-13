from params.base import DecoderDGP, Decoder
import torch.nn as nn


class MLE(nn.Module):
    def __init__(self, m, n, decoder_dgp, fit_s2):
        super(MLE, self).__init__()
        self.name, self.m, self.n = "mle", m, n
        self.fit_s2 = fit_s2
        if decoder_dgp:
            self.decoder = DecoderDGP(m, n, fit_s2=self.fit_s2)
        else:
            self.decoder = Decoder(m, n, fit_s2=self.fit_s2)

    def forward(self, z):
        mean = self.decoder(z)

        return mean
