#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchvision import transforms

from conv_onet.Data.field.index_field import IndexField
from conv_onet.Data.field.partial_point_cloud_field import PartialPointCloudField
from conv_onet.Data.field.patch_point_cloud_field import PatchPointCloudField
from conv_onet.Data.field.patch_points_field import PatchPointsField
from conv_onet.Data.field.point_cloud_field import PointCloudField
from conv_onet.Data.field.points_field import PointsField
from conv_onet.Data.field.voxels_field import VoxelsField

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

    input_type = cfg['data']['input_type']
    fields = {}
    if cfg['data']['points_file'] is not None:
        if input_type != 'pointcloud_crop':
            fields['points'] = PointsField(
                cfg['data']['points_file'],
                points_transform,
                unpackbits=cfg['data']['points_unpackbits'],
                multi_files=cfg['data']['multi_files'])
        else:
            fields['points'] = PatchPointsField(
                cfg['data']['points_file'],
                transform=points_transform,
                unpackbits=cfg['data']['points_unpackbits'],
                multi_files=cfg['data']['multi_files'])

    if mode in ('val', 'test'):
        points_iou_file = cfg['data']['points_iou_file']
        voxels_file = cfg['data']['voxels_file']
        if points_iou_file is not None:
            if input_type == 'pointcloud_crop':
                fields['points_iou'] = PatchPointsField(
                    points_iou_file,
                    unpackbits=cfg['data']['points_unpackbits'],
                    multi_files=cfg['data']['multi_files'])
            else:
                fields['points_iou'] = PointsField(
                    points_iou_file,
                    unpackbits=cfg['data']['points_unpackbits'],
                    multi_files=cfg['data']['multi_files'])
        if voxels_file is not None:
            fields['voxels'] = VoxelsField(voxels_file)
    return fields


def get_inputs_field(mode, cfg):
    ''' Returns the inputs fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): config dictionary
    '''
    input_type = cfg['data']['input_type']

    if input_type is None:
        inputs_field = None
    elif input_type == 'pointcloud':
        transform = transforms.Compose([
            SubsamplePointcloud(cfg['data']['pointcloud_n']),
            PointcloudNoise(cfg['data']['pointcloud_noise'])
        ])
        inputs_field = PointCloudField(cfg['data']['pointcloud_file'],
                                       transform,
                                       multi_files=cfg['data']['multi_files'])
    elif input_type == 'partial_pointcloud':
        transform = transforms.Compose([
            SubsamplePointcloud(cfg['data']['pointcloud_n']),
            PointcloudNoise(cfg['data']['pointcloud_noise'])
        ])
        inputs_field = PartialPointCloudField(
            cfg['data']['pointcloud_file'],
            transform,
            multi_files=cfg['data']['multi_files'])
    elif input_type == 'pointcloud_crop':
        transform = transforms.Compose([
            SubsamplePointcloud(cfg['data']['pointcloud_n']),
            PointcloudNoise(cfg['data']['pointcloud_noise'])
        ])

        inputs_field = PatchPointCloudField(
            cfg['data']['pointcloud_file'],
            transform,
            multi_files=cfg['data']['multi_files'],
        )

    elif input_type == 'voxels':
        inputs_field = VoxelsField(cfg['data']['voxels_file'])
    elif input_type == 'idx':
        inputs_field = IndexField()
    else:
        raise ValueError('Invalid input type (%s)' % input_type)
    return inputs_field
