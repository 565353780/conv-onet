#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F

from conv_onet.Model.layer.resnet_block_fc import ResnetBlockFC

from conv_onet.Method.common import map2local


class PatchLocalDecoder(nn.Module):
    ''' Decoder adapted for crop training.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        local_coord (bool): whether to use local coordinate
        unit_size (float): defined voxel unit size for local system
        pos_encoding (str): method for the positional encoding, linear|sin_cos
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]

    '''

    def __init__(self,
                 dim=3,
                 c_dim=128,
                 hidden_size=256,
                 leaky=False,
                 n_blocks=5,
                 sample_mode='bilinear',
                 local_coord=False,
                 pos_encoding='linear',
                 unit_size=0.1,
                 padding=0.1):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if c_dim != 0:
            self.fc_c = nn.ModuleList(
                [nn.Linear(c_dim, hidden_size) for _ in range(n_blocks)])

        #self.fc_p = nn.Linear(dim, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.blocks = nn.ModuleList(
            [ResnetBlockFC(hidden_size) for _ in range(n_blocks)])

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode

        if local_coord:
            self.map2local = map2local(unit_size, pos_encoding=pos_encoding)
        else:
            self.map2local = None

        if pos_encoding == 'sin_cos':
            self.fc_p = nn.Linear(60, hidden_size)
        else:
            self.fc_p = nn.Linear(dim, hidden_size)

    def sample_feature(self, xy, c, fea_type='2d'):
        if fea_type == '2d':
            xy = xy[:, :, None].float()
            vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
            c = F.grid_sample(c,
                              vgrid,
                              padding_mode='border',
                              align_corners=True,
                              mode=self.sample_mode).squeeze(-1)
        else:
            xy = xy[:, :, None, None].float()
            vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
            c = F.grid_sample(c,
                              vgrid,
                              padding_mode='border',
                              align_corners=True,
                              mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def forward(self, p, c_plane):
        p_n = p['p_n']
        p = p['p']

        if self.c_dim != 0:
            c = self.sample_feature(p_n['grid'],
                                    c_plane['grid'],
                                    fea_type='3d')
            c = c.transpose(1, 2)

        p = p.float()
        if self.map2local:
            p = self.map2local(p)

        net = self.fc_p(p)
        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)
            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)
        return out
