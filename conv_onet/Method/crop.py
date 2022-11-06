#!/usr/bin/env python
# -*- coding: utf-8 -*-


def getFeaturePoints(crop, feature_name):
    assert feature_name in crop.feature_dict.keys()

    feature = crop.feature_dict[feature_name]

    feature_shape = feature.shape

    assert len(feature_shape) == 3

    min_point = crop.bbox.min_point
    diff_point = crop.bbox.diff_point

    x_diff = diff_point.x / feature_shape[0]
    y_diff = diff_point.y / feature_shape[1]
    z_diff = diff_point.z / feature_shape[2]

    x_start = min_point.x - 0.5 * diff_point.x
    y_start = min_point.y - 0.5 * diff_point.y
    z_start = min_point.z - 0.5 * diff_point.z

    feature_points = []
    features = []

    for i in range(feature_shape[0]):
        point_x = x_start + i * x_diff
        for j in range(feature_shape[1]):
            point_y = y_start + j * y_diff
            for k in range(feature_shape[2]):
                point_z = z_start + k * z_diff

                point = [point_x, point_y, point_z]
                feature_points.append(point)
                features.append(feature[i][j][k])
    return feature_points, features


def getFeaturePointsByIdx(crop_space, space_idx, feature_name):
    i, j, k = space_idx
    crop = crop_space.space[i][j][k]
    return getFeaturePoints(crop, feature_name)


def getFeaturePointsByIdxWithPool(inputs):
    crop_space, space_idx, feature_name = inputs
    return getFeaturePointsByIdx(crop_space, space_idx, feature_name)
