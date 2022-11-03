#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max

from conv_onet.Model.layer.resnet_block_fc import ResnetBlockFC
from conv_onet.Model.encoder.unet3d.unet3d import UNet3D

from conv_onet.Method.common import map2local


class PatchLocalPoolPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.
        First transform input points to local system based on the given voxel size.
        Support non-fixed number of point cloud, but need to precompute the index

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
        local_coord (bool): whether to use local coordinate
        pos_encoding (str): method for the positional encoding, linear|sin_cos
        unit_size (float): defined voxel unit size for local system
    '''

    def __init__(self,
                 c_dim=128,
                 dim=3,
                 hidden_dim=128,
                 scatter_type='max',
                 plane_resolution=None,
                 grid_resolution=None,
                 padding=0.1,
                 n_blocks=5,
                 local_coord=False,
                 pos_encoding='linear',
                 unit_size=0.1):
        super().__init__()
        self.c_dim = c_dim

        self.blocks = nn.ModuleList([
            ResnetBlockFC(2 * hidden_dim, hidden_dim) for _ in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim
        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.padding = padding

        self.unet3d = UNet3D(32, 32, f_maps=32)

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')

        if local_coord:
            self.map2local = map2local(unit_size, pos_encoding=pos_encoding)
        else:
            self.map2local = None

        if pos_encoding == 'sin_cos':
            self.fc_pos = nn.Linear(60, 2 * hidden_dim)
        else:
            self.fc_pos = nn.Linear(dim, 2 * hidden_dim)

    def generate_plane_features(self, index, c):
        c = c.permute(0, 2, 1)
        # scatter plane features from points
        if index.max() < self.reso_plane**2:
            fea_plane = c.new_zeros(c.size(0), self.c_dim, self.reso_plane**2)
            fea_plane = scatter_mean(c, index,
                                     out=fea_plane)  # B x c_dim x reso^2
        else:
            fea_plane = scatter_mean(c, index)  # B x c_dim x reso^2
            if fea_plane.shape[-1] > self.reso_plane**2:  # deal with outliers
                fea_plane = fea_plane[:, :, :-1]

        fea_plane = fea_plane.reshape(c.size(0), self.c_dim, self.reso_plane,
                                      self.reso_plane)
        return fea_plane

    def generate_grid_features(self, index, c):
        # scatter grid features from points
        c = c.permute(0, 2, 1)
        if index.max() < self.reso_grid**3:
            fea_grid = c.new_zeros(c.size(0), self.c_dim, self.reso_grid**3)
            fea_grid = scatter_mean(c, index,
                                    out=fea_grid)  # B x c_dim x reso^3
        else:
            fea_grid = scatter_mean(c, index)  # B x c_dim x reso^3
            if fea_grid.shape[-1] > self.reso_grid**3:  # deal with outliers
                fea_grid = fea_grid[:, :, :-1]
        fea_grid = fea_grid.reshape(c.size(0), self.c_dim, self.reso_grid,
                                    self.reso_grid, self.reso_grid)
        fea_grid = self.unet3d(fea_grid)
        return fea_grid

    def pool_local(self, index, c):
        _, fea_dim = c.size(0), c.size(2)

        c_out = 0
        # scatter plane features from points
        fea = self.scatter(c.permute(0, 2, 1), index['grid'])
        if self.scatter == scatter_max:
            fea = fea[0]
        # gather feature back to points
        fea = fea.gather(dim=2, index=index['grid'].expand(-1, fea_dim, -1))
        c_out += fea
        return c_out.permute(0, 2, 1)

    def forward(self, inputs):
        p = inputs['points']
        index = inputs['index']

        if self.map2local:
            pp = self.map2local(p)
            net = self.fc_pos(pp)
        else:
            net = self.fc_pos(p)

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        c = self.fc_c(net)

        fea = {}
        fea['grid'] = self.generate_grid_features(index['grid'], c)
        return fea
