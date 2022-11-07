#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm

from conv_onet.Config.crop import UNIT_LB, UNIT_UB

from conv_onet.Data.crop import Crop


class CropSpace(object):

    def __init__(self, input_crop_size, query_crop_size):
        self.input_crop_size = input_crop_size
        self.query_crop_size = query_crop_size

        self.space_size = None
        self.space_idx_list = None
        self.space = None

        self.point_array = None

        self.createSpace()
        return

    def resetSpace(self):
        if self.space is None:
            return True

        assert self.space_size is not None

        for i in range(self.space_size[0]):
            for j in range(self.space_size[1]):
                for k in range(self.space_size[2]):
                    self.space[i][j][k].reset()
        return True

    def reset(self):
        self.point_array = None
        return True

    def getCropNum(self):
        if self.space_size is None:
            return 0
        return np.prod(self.space_size)

    def getCrop(self, i, j, k):
        assert self.space_size is not None
        assert self.space is not None

        assert 0 <= i < self.space_size[0]
        assert 0 <= j < self.space_size[1]
        assert 0 <= k < self.space_size[2]

        return self.space[i][j][k]

    def getCropByIdx(self, crop_idx):
        assert self.space_size is not None
        assert self.space_idx_list is not None
        assert self.space is not None
        assert crop_idx < len(self.space_idx_list)

        i, j, k = self.space_idx_list[crop_idx]
        return self.getCrop(i, j, k)

    def createSpace(self):
        self.space_size = np.ceil(
            (UNIT_UB - UNIT_LB) / self.query_crop_size).astype(int)

        self.space = np.zeros(self.space_size).tolist()

        self.space_idx_list = []

        for i in range(self.space_size[0]):
            min_point_x = UNIT_LB[0] + i * self.query_crop_size
            for j in range(self.space_size[1]):
                min_point_y = UNIT_LB[1] + j * self.query_crop_size
                for k in range(self.space_size[2]):
                    min_point_z = UNIT_LB[2] + k * self.query_crop_size

                    min_point = np.array(
                        [min_point_x, min_point_y, min_point_z])
                    max_point = min_point + self.query_crop_size

                    center = (min_point + max_point) / 2.0

                    input_min_point = center - self.input_crop_size / 2.0
                    input_max_point = center + self.input_crop_size / 2.0

                    self.space[i][j][k] = Crop.fromList(
                        center, [min_point.tolist(),
                                 max_point.tolist()],
                        [input_min_point.tolist(),
                         input_max_point.tolist()])

                    self.space_idx_list.append([i, j, k])
        return True

    def updatePointArray(self, point_array):
        assert self.space_size is not None
        assert self.space_idx_list is not None
        assert self.space is not None

        self.point_array = point_array

        for i, j, k in self.space_idx_list:
            crop = self.getCrop(i, j, k)
            mask_x = (self.point_array[:, :, 0] >= crop.input_bbox.min_point.x) &\
                (self.point_array[:, :, 0] <
                 crop.input_bbox.max_point.x)
            mask_y = (self.point_array[:, :, 1] >= crop.input_bbox.min_point.y) &\
                (self.point_array[:, :, 1] <
                 crop.input_bbox.max_point.y)
            mask_z = (self.point_array[:, :, 2] >= crop.input_bbox.min_point.z) &\
                (self.point_array[:, :, 2] <
                 crop.input_bbox.max_point.z)
            mask = mask_x & mask_y & mask_z
            mask_point_array = self.point_array[mask]
            crop.updateInputPointArray(mask_point_array)
        return True

    def updateCropFeature(self, feature_name, func, print_progress=False):
        assert self.space_size is not None
        assert self.space_idx_list is not None
        assert self.space is not None

        for_data = self.space_idx_list
        if print_progress:
            print("[INFO][CropSpace::updateCropFeature]")
            print("\t start update crop feature for " + feature_name + "...")
            for_data = tqdm(for_data)
        for i, j, k in for_data:
            crop = self.getCrop(i, j, k)
            crop.updateFeature(feature_name, func(crop))
        return True

    def updateCropFeatureDict(self, func, print_progress=False):
        assert self.space_size is not None
        assert self.space_idx_list is not None
        assert self.space is not None

        for_data = self.space_idx_list
        if print_progress:
            print("[INFO][CropSpace::updateCropFeatureDict]")
            print("\t start update crop feature...")
            for_data = tqdm(for_data)
        for i, j, k in for_data:
            crop = self.getCrop(i, j, k)
            crop.updateFeatureDict(func(crop))
        return True

    def getFeatureMask(self, feature_name):
        assert self.space_size is not None
        assert self.space_idx_list is not None
        assert self.space is not None

        feature_mask = np.zeros(self.space_size, dtype=bool)
        for i, j, k in self.space_idx_list:
            feature_mask[i][j][k] = self.getCrop(
                i, j, k).feature_dict[feature_name] is not None
        return feature_mask

    def getMaskFeatureIdxArray(self, feature_name):
        feature_mask = self.getFeatureMask(feature_name)
        return np.dstack(np.where(feature_mask == True))[0]
