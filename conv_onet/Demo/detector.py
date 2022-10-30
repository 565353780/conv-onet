#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d

from conv_onet.Module.detector import Detector


def demo():
    pcd_file_path = \
        "/home/chli/chLi/ScanNet/objects/scene0474_02/2_chair.ply"

    pcd = o3d.io.read_point_cloud(pcd_file_path)
    points = np.array(pcd.points)
    point_idx_array = np.arange(0, points.shape[0])
    sample_point_idx_array = np.random.choice(point_idx_array,
                                              int(points.shape[0] / 2),
                                              replace=False)
    sample_points = points[sample_point_idx_array]

    detector = Detector()
    detector.detectPointArray(sample_points)
    return True
