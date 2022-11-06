#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from multiprocessing import Pool

from conv_onet.Method.crop import getFeaturePointsByIdxWithPool


def renderCropSpaceFeature(crop_space, feature_name):
    pcd = o3d.geometry.PointCloud()

    points = []
    features = []

    inputs_list = []
    for space_idx in crop_space.space_idx_list:
        inputs_list.append([crop_space, space_idx, feature_name])

    with Pool(os.cpu_count()) as pool:
        result = list(
            tqdm(pool.imap(getFeaturePointsByIdxWithPool, inputs_list),
                 total=len(inputs_list)))
    for feature_points, features in result:
        points.append(feature_points)
        features.append(features)

    points = np.array(points).reshape(-1, 3)
    features = np.array(features).reshape(-1, 3)
    print(points.shape)
    print(features.shape)

    max_features = np.max(features)
    features = features / max_features

    colors = np.zeros_like(points)
    colors[:, 0] = features
    colors[:, 2] = 1 - features

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])
    return True
