#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchvision import transforms

from conv_onet.Data.field.patch_point_cloud_field import PatchPointCloudField

from conv_onet.Data.transform.pointcloud_noise import PointcloudNoise
from conv_onet.Data.transform.subsample_pointcloud import SubsamplePointcloud


def get_inputs_field(pointcloud_n, pointcloud_noise, pointcloud_file):
    transform = transforms.Compose(
        [SubsamplePointcloud(pointcloud_n),
         PointcloudNoise(pointcloud_noise)])

    inputs_field = PatchPointCloudField(pointcloud_file, transform)
    return inputs_field
