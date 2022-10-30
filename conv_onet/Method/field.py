#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchvision import transforms

from conv_onet.Data.field.patch_point_cloud_field import PatchPointCloudField
from conv_onet.Data.field.patch_points_field import PatchPointsField

from conv_onet.Data.transform.pointcloud_noise import PointcloudNoise
from conv_onet.Data.transform.subsample_pointcloud import SubsamplePointcloud
from conv_onet.Data.transform.subsample_points import SubsamplePoints


def get_data_fields(mode, cfg):
    ''' Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    points_transform = SubsamplePoints(cfg['data']['points_subsample'])

    fields = {}
    if cfg['data']['points_file'] is not None:
        fields['points'] = PatchPointsField(
            cfg['data']['points_file'],
            transform=points_transform,
            unpackbits=True)

    if mode in ('val', 'test'):
        points_iou_file = cfg['data']['points_iou_file']
        if points_iou_file is not None:
            fields['points_iou'] = PatchPointsField(
                points_iou_file,
                unpackbits=True)
    return fields


def get_inputs_field(pointcloud_n, pointcloud_noise, pointcloud_file):
    transform = transforms.Compose(
        [SubsamplePointcloud(pointcloud_n),
         PointcloudNoise(pointcloud_noise)])

    inputs_field = PatchPointCloudField(pointcloud_file, transform)
    return inputs_field
