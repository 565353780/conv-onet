#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
from torch import distributions as dist

from conv_onet.Model.decoder.patch_local_decoder import PatchLocalDecoder

from conv_onet.Model.encoder.pointnet.patch_local_pool_pointnet import PatchLocalPoolPointnet

from conv_onet.Method.common import update_reso


class ConvolutionalOccupancyNetwork(nn.Module):

    def __init__(self, padding, unit_size, query_vol_size, device):
        super().__init__()

        self.decoder = PatchLocalDecoder(c_dim=32,
                                         hidden_size=32,
                                         local_coord=True,
                                         unit_size=unit_size).to(device)

        reso = query_vol_size + 2**6 - 1
        grid_resolution = update_reso(reso)
        self.encoder = PatchLocalPoolPointnet(c_dim=32,
                                              hidden_dim=32,
                                              grid_resolution=grid_resolution,
                                              padding=padding,
                                              local_coord=True,
                                              unit_size=unit_size)

        self._device = device
        return

    @classmethod
    def fromConfig(cls, cfg, device=None):
        # update the feature volume/plane resolution

        return cls(cfg['data']['padding'], cfg['data']['unit_size'],
                   cfg['data']['query_vol_size'], device)

    def to(self, device):
        model = super().to(device)
        model._device = device
        return model

    def encode_inputs(self, inputs):
        return self.encoder(inputs)

    def decode(self, p, c, **kwargs):
        logits = self.decoder(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def forward(self, p, inputs, **kwargs):
        c = self.encode_inputs(inputs)
        p_r = self.decode(p, c, **kwargs)
        return p_r
