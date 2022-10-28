#!/usr/bin/env python
# -*- coding: utf-8 -*-

from conv_onet.Model.encoder.unet3d.ext_resnet_block import ExtResNetBlock
from conv_onet.Model.encoder.unet3d.abstract_3d_unet import Abstract3DUNet


class ResidualUNet3D(Abstract3DUNet):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ExtResNetBlock as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
    Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 final_sigmoid=True,
                 f_maps=64,
                 layer_order='gcr',
                 num_groups=8,
                 num_levels=5,
                 is_segmentation=True,
                 **kwargs):
        super(ResidualUNet3D, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             final_sigmoid=final_sigmoid,
                                             basic_module=ExtResNetBlock,
                                             f_maps=f_maps,
                                             layer_order=layer_order,
                                             num_groups=num_groups,
                                             num_levels=num_levels,
                                             is_segmentation=is_segmentation,
                                             **kwargs)
