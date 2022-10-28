#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn


def conv3d(in_channels, out_channels, kernel_size, bias, padding=1):
    return nn.Conv3d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=padding,
                     bias=bias)
