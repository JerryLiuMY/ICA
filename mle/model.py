from params.base import DecoderDGP, Decoder
import torch.nn as nn


class MLE(nn.Module):
    def __init__(self, m, n, fit_s2, decoder_info):
        super(MLE, self).__init__()
        self.name, self.m, self.n, self.fit_s2 = "mle", m, n, fit_s2
        self.decoder_dgp, self.activation_name = decoder_info
        if self.decoder_dgp:
            self.decoder = DecoderDGP(m, n, fit_s2=self.fit_s2, activation_name=self.activation_name)
        else:
            self.decoder = Decoder(m, n, fit_s2=self.fit_s2)

    def forward(self, z):
        mean = self.decoder(z)

        return mean
