#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchvision import transforms

from conv_onet.Data.field.patch_point_cloud_field import PatchPointCloudField
from conv_onet.Data.field.patch_points_field import PatchPointsField
from conv_onet.Data.field.voxels_field import VoxelsField

from conv_onet.Data.transform.pointcloud_noise import PointcloudNoise
from conv_onet.Data.transform.subsample_pointcloud import SubsamplePointcloud
from conv_onet.Data.transform.subsample_points import SubsamplePoints


def get_inputs_field(mode, cfg):
    ''' Returns the inputs fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): config dictionary
    '''
    transform = transforms.Compose([
        SubsamplePointcloud(cfg['data']['pointcloud_n']),
        PointcloudNoise(cfg['data']['pointcloud_noise'])
    ])

    inputs_field = PatchPointCloudField(
        cfg['data']['pointcloud_file'],
        transform,
        multi_files=cfg['data']['multi_files'],
    )
    return inputs_field
