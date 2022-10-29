#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from conv_onet.Module.detector import Detector


def demo():
    model_path = \
        "/home/chli/chLi/conv-onet/demo_data/demo/Matterport3D_processed/17DRP5sb8fy/pointcloud.npz"
    pointcloud_dict = np.load(model_path)

    points = pointcloud_dict['points']
    point_idx_array = np.arange(0, points.shape[0])
    sample_point_idx_array = np.random.choice(point_idx_array,
                                              int(points.shape[0] / 2),
                                              replace=False)
    sample_points = points[sample_point_idx_array]

    detector = Detector()
    #  detector.detectPointArray(sample_points)
    detector.detectAll()
    return True
