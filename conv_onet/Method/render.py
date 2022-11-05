#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d



def renderCropSpaceFeature(crop_space, feature_name):
    pcd = o3d.geometry.PointCloud()

    points = []
    features = []

    for i, j, k in crop_space.space_idx_list:
        points.append(crop_space.space[i][j][k].center.toList())
        features.append(crop_space.space[i][j][k].center.toList())
    return True
