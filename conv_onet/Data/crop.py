#!/usr/bin/env python
# -*- coding: utf-8 -*-

from conv_onet.Data.point import Point
from conv_onet.Data.bbox import BBox


class Crop(object):

    def __init__(self, center=Point(), bbox=BBox(), input_bbox=BBox()):
        self.center = center
        self.bbox = bbox
        self.input_bbox = input_bbox

        self.input_point_array = None
        self.feature = None
        return

    @classmethod
    def fromList(cls,
                 center_list=[0.0, 0.0, 0.0],
                 bbox_list=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                 input_bbox_list=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]):
        return cls(Point.fromList(center_list), BBox.fromList(bbox_list),
                   BBox.fromList(input_bbox_list))

    def updateInputPointArray(self, input_point_array):
        self.input_point_array = input_point_array
        return True

    def updateFeature(self, feature):
        self.feature = feature
        return True
