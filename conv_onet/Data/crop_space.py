#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from conv_onet.Config.crop import UNIT_LB, UNIT_UB

from conv_onet.Data.crop import Crop


class CropSpace(object):

    def __init__(self, input_crop_size, query_crop_size):
        self.input_crop_size = input_crop_size
        self.query_crop_size = query_crop_size

        self.point_array = None
        self.space_size = None
        self.space = None

        self.createSpace()
        return

    def reset(self):
        self.point_array = None
        self.space_size = None
        self.space = None
        return True

    def createSpace(self):
        self.space_size = np.ceil(
            (UNIT_UB - UNIT_LB) / self.query_crop_size).astype(int)

        self.space = np.zeros(self.space_size).tolist()

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
        return True

    def updatePointArray(self, point_array):
        assert self.space_size is not None
        assert self.space is not None

        self.point_array = point_array

        for i in range(self.space_size[0]):
            for j in range(self.space_size[1]):
                for k in range(self.space_size[2]):
                    crop = self.space[i][j][k]
                    mask_x = (self.point_array[:, :, 0] >= crop.input_bbox.min_point.x) &\
                            (self.point_array[:, :, 0] < crop.input_bbox.max_point.x)
                    mask_y = (self.point_array[:, :, 1] >= crop.input_bbox.min_point.y) &\
                            (self.point_array[:, :, 1] < crop.input_bbox.max_point.y)
                    mask_z = (self.point_array[:, :, 2] >= crop.input_bbox.min_point.z) &\
                            (self.point_array[:, :, 2] < crop.input_bbox.max_point.z)
                    mask = mask_x & mask_y & mask_z
                    mask_point_array = self.point_array[mask]
                    crop.updateInputPointArray(mask_point_array)
        return True

    def getCropNum(self):
        if self.space_size is None:
            return 0
        return np.sum(self.space_size)
