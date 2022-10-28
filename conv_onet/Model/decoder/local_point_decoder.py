#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from conv_onet.Model.layer.resnet_block_fc import ResnetBlockFC


class LocalPointDecoder(nn.Module):
    ''' Decoder for PointConv Baseline.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        n_blocks (int): number of blocks ResNetBlockFC layers
        sample_mode (str): sampling mode  for points
    '''

    def __init__(self,
                 dim=3,
                 c_dim=128,
                 hidden_size=256,
                 leaky=False,
                 n_blocks=5,
                 sample_mode='gaussian',
                 **kwargs):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if c_dim != 0:
            self.fc_c = nn.ModuleList(
                [nn.Linear(c_dim, hidden_size) for i in range(n_blocks)])

        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(hidden_size) for i in range(n_blocks)])

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        if sample_mode == 'gaussian':
            self.var = kwargs['gaussian_val']**2

    def sample_point_feature(self, q, p, fea):
        # q: B x M x 3
        # p: B x N x 3
        # fea: B x N x c_dim
        #p, fea = c
        if self.sample_mode == 'gaussian':
            # distance betweeen each query point to the point cloud
            dist = -((p.unsqueeze(1).expand(-1, q.size(1), -1, -1) -
                      q.unsqueeze(2)).norm(dim=3) + 10e-6)**2
            weight = (dist / self.var).exp()  # Guassian kernel
        else:
            weight = 1 / ((p.unsqueeze(1).expand(-1, q.size(1), -1, -1) -
                           q.unsqueeze(2)).norm(dim=3) + 10e-6)

        #weight normalization
        weight = weight / weight.sum(dim=2).unsqueeze(-1)

        c_out = weight @ fea  # B x M x c_dim

        return c_out

    def forward(self, p, c, **kwargs):
        n_points = p.shape[1]

        if n_points >= 30000:
            pp, fea = c
            c_list = []
            for p_split in torch.split(p, 10000, dim=1):
                if self.c_dim != 0:
                    c_list.append(self.sample_point_feature(p_split, pp, fea))
            c = torch.cat(c_list, dim=1)

        else:
            if self.c_dim != 0:
                pp, fea = c
                c = self.sample_point_feature(p, pp, fea)

        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out
