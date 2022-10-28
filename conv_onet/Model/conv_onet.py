#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch import distributions as dist


class ConvolutionalOccupancyNetwork(nn.Module):

    def __init__(self, decoder, encoder=None, device=None):
        super().__init__()

        self.decoder = decoder.to(device)

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device

    def forward(self, p, inputs, sample=True, **kwargs):
        c = self.encode_inputs(inputs)
        p_r = self.decode(p, c, **kwargs)
        return p_r

    def encode_inputs(self, inputs):
        assert self.encoder is not None

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            #FIXME: Return inputs?
            c = torch.empty(inputs.size(0), 0)
        return c

    def decode(self, p, c, **kwargs):
        logits = self.decoder(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def to(self, device):
        model = super().to(device)
        model._device = device
        return model
