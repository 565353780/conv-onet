#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from multiprocessing import Pool

from conv_onet.Method.crop import \
    getFeaturePoints, getFeaturePointsByIdxWithPool


def renderCropSpaceFeature(crop_space,
                           feature_name,
                           merge_num=1,
                           print_progress=False,
                           with_pool=False):
    pcd = o3d.geometry.PointCloud()

    points = []
    features = []

    mask_feature_idx_array = crop_space.getMaskFeatureIdxArray('occ')
    inputs_list = []
    for mask_feature_idx in mask_feature_idx_array:
        inputs_list.append(
            [crop_space, mask_feature_idx, feature_name, merge_num])

    if with_pool:
        if print_progress:
            print("[INFO][render::renderCropSpaceFeature]")
            print("\t start collect feature points with pool...")
            print("[WARN][render::renderCropSpaceFeature]")
            print("\t collect with pool may cause RAM out of memory!")
            print("\t if process not running, please stop it!")
            with Pool(os.cpu_count()) as pool:
                result = list(
                    tqdm(pool.imap(getFeaturePointsByIdxWithPool, inputs_list),
                         total=len(inputs_list)))
        else:
            with Pool(os.cpu_count()) as pool:
                result = list(
                    pool.imap(getFeaturePointsByIdxWithPool, inputs_list))
        for feature_points, curr_features in result:
            points.append(feature_points)
            features.append(curr_features)
    else:
        for_data = mask_feature_idx_array
        if print_progress:
            print("[INFO][render::renderCropSpaceFeature]")
            print("\t start collect feature points...")
            for_data = tqdm(for_data)
        for i, j, k in for_data:
            crop = crop_space.getCrop(i, j, k)
            feature_points, curr_features = getFeaturePoints(
                crop, feature_name, merge_num)
            points.append(feature_points)
            features.append(curr_features)

    points = np.array(points).reshape(-1, 3)
    features = np.array(features).reshape(-1)

    valid_feature_idx = np.where(features > 0)[0]
    points = points[valid_feature_idx]
    features = features[valid_feature_idx]

    max_features = np.max(features)
    features = features / max_features

    colors = np.zeros_like(points)
    colors[:, 0] = features
    colors[:, 2] = 1 - features

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])
    return True
