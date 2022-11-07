#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from math import ceil


def getFeaturePoints(crop, feature_name, merge_num=1):
    assert feature_name in crop.feature_dict.keys()

    feature = crop.feature_dict[feature_name]

    feature_shape = feature.shape

    merge_feature_shape = [
        1.0 * feature_shape[i] / merge_num for i in range(3)
    ]
    ceil_merge_feature_shape = [ceil(merge_feature_shape[i]) for i in range(3)]

    assert len(feature_shape) == 3

    min_point = crop.bbox.min_point
    diff_point = crop.bbox.diff_point

    x_diff = 1.0 * diff_point.x / merge_feature_shape[0]
    y_diff = 1.0 * diff_point.y / merge_feature_shape[1]
    z_diff = 1.0 * diff_point.z / merge_feature_shape[2]

    x_start = min_point.x - 0.5 * x_diff
    y_start = min_point.y - 0.5 * y_diff
    z_start = min_point.z - 0.5 * z_diff

    feature_points = []
    features = []

    for i in range(ceil_merge_feature_shape[0]):
        point_x = x_start + i * x_diff
        feature_x_start = i * merge_num
        feature_x_end = min(feature_x_start + merge_num, feature_shape[0])
        for j in range(ceil_merge_feature_shape[1]):
            point_y = y_start + j * y_diff
            feature_y_start = j * merge_num
            feature_y_end = min(feature_y_start + merge_num, feature_shape[1])
            for k in range(ceil_merge_feature_shape[2]):
                point_z = z_start + k * z_diff
                feature_z_start = k * merge_num
                feature_z_end = min(feature_z_start + merge_num,
                                    feature_shape[2])

                point = [point_x, point_y, point_z]

                merge_feature_array = feature[feature_x_start:feature_x_end,
                                              feature_y_start:feature_y_end,
                                              feature_z_start:feature_z_end]

                merge_feature = np.max(merge_feature_array)

                feature_points.append(point)
                features.append(merge_feature)
    return feature_points, features


def getFeaturePointsByIdx(crop_space, space_idx, feature_name, merge_num=1):
    i, j, k = space_idx
    crop = crop_space.space[i][j][k]
    return getFeaturePoints(crop, feature_name, merge_num)


def getFeaturePointsByIdxWithPool(inputs):
    assert 3 <= len(inputs) <= 4

    if len(inputs) == 3:
        crop_space, space_idx, feature_name = inputs
        merge_num = 1
    else:
        crop_space, space_idx, feature_name, merge_num = inputs

    return getFeaturePointsByIdx(crop_space, space_idx, feature_name,
                                 merge_num)
