#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from conv_onet.Data.point import Point
from conv_onet.Data.bbox import BBox


class Crop(object):

    def __init__(self, center=Point(), bbox=BBox(), input_bbox=BBox()):
        self.center = center
        self.bbox = bbox
        self.input_bbox = input_bbox

        self.input_point_array = None
        self.feature_dict = {}
        return

    def reset(self):
        self.input_point_array = None
        self.feature_dict = {}
        return True

    @classmethod
    def fromList(cls,
                 center_list=[0.0, 0.0, 0.0],
                 bbox_list=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                 input_bbox_list=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]):
        return cls(Point.fromList(center_list), BBox.fromList(bbox_list),
                   BBox.fromList(input_bbox_list))

    def isEmpty(self):
        if self.input_point_array is None:
            return True
        if self.input_point_array.shape[0] == 0:
            return True
        return False

    def updateInputPointArray(self, input_point_array):
        self.input_point_array = input_point_array

        if self.isEmpty():
            self.feature_dict['valid'] = None
        else:
            self.feature_dict['valid'] = np.array([1.0], dtype=float)
        return True

    def updateFeature(self, feature_name, feature_value):
        self.feature_dict[feature_name] = feature_value
        return True

    def updateFeatureDict(self, feature_dict):
        self.feature_dict.update(feature_dict)
        return True
