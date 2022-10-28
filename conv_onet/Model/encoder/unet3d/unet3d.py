#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import importlib

from conv_onet.Model.encoder.unet3d.double_conv import DoubleConv
from conv_onet.Model.encoder.unet3d.abstract_3d_unet import Abstract3DUNet


class UNet3D(Abstract3DUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 final_sigmoid=True,
                 f_maps=64,
                 layer_order='gcr',
                 num_groups=8,
                 num_levels=4,
                 is_segmentation=True,
                 **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     **kwargs)


def get_model(config):

    def _model_class(class_name):
        m = importlib.import_module('pytorch3dunet.unet3d.model')
        clazz = getattr(m, class_name)
        return clazz

    assert 'model' in config, 'Could not find model configuration'
    model_config = config['model']
    model_class = _model_class(model_config['name'])
    return model_class(**model_config)


if __name__ == "__main__":
    """
    testing
    """
    in_channels = 1
    out_channels = 1
    f_maps = 32
    num_levels = 3
    model = UNet3D(in_channels,
                   out_channels,
                   f_maps=f_maps,
                   num_levels=num_levels,
                   layer_order='cr')
    print(model)

    reso = 42

    import numpy as np
    import torch
    x = np.zeros((1, 1, reso, reso, reso))
    x[:, :, int(reso / 2 - 1), int(reso / 2 - 1), int(reso / 2 - 1)] = np.nan
    x = torch.FloatTensor(x)

    out = model(x)
    print('%f' % (torch.sum(torch.isnan(out)).detach().cpu().numpy() /
                  (reso * reso * reso)))
