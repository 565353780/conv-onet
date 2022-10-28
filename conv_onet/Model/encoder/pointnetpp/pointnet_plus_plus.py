#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn

from conv_onet.Model.encoder.pointnetpp.pointnet_set_abstraction import PointNetSetAbstraction
from conv_onet.Model.encoder.pointnetpp.pointnet_feature_propagation import PointNetFeaturePropagation


class PointNetPlusPlus(nn.Module):

    def __init__(self, dim=None, c_dim=128, padding=0.1):
        super(PointNetPlusPlus, self).__init__()

        self.sa1 = PointNetSetAbstraction(npoint=512,
                                          radius=0.2,
                                          nsample=32,
                                          in_channel=6,
                                          mlp=[64, 64, 128],
                                          group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128,
                                          radius=0.4,
                                          nsample=64,
                                          in_channel=128 + 3,
                                          mlp=[128, 128, 256],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None,
                                          radius=None,
                                          nsample=None,
                                          in_channel=256 + 3,
                                          mlp=[256, 512, 1024],
                                          group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128,
                                              mlp=[128, 128, c_dim])

    def forward(self, xyz):
        xyz = xyz.permute(0, 2, 1)
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        return xyz.permute(0, 2, 1), l0_points.permute(0, 2, 1)
