#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from conv_onet.Method.voxel_grid import check_voxel_boundary, check_voxel_occupied
from conv_onet.Method.mesh import upsample3d_nn


class MultiGridExtractor(object):

    def __init__(self, resolution0, threshold):
        # Attributes
        self.resolution = resolution0
        self.threshold = threshold

        # Voxels are active or inactive,
        # values live on the space between voxels and are either
        # known exactly or guessed by interpolation (unknown)
        shape_voxels = (resolution0, ) * 3
        shape_values = (resolution0 + 1, ) * 3
        self.values = np.empty(shape_values)
        self.value_known = np.full(shape_values, False)
        self.voxel_active = np.full(shape_voxels, True)

    def query(self):
        # Query locations in grid that are active but unkown
        idx1, idx2, idx3 = np.where(~self.value_known & self.value_active)
        points = np.stack([idx1, idx2, idx3], axis=-1)
        return points

    def update(self, points, values):
        # Update locations and set known status to true
        idx0, idx1, idx2 = points.transpose()
        self.values[idx0, idx1, idx2] = values
        self.value_known[idx0, idx1, idx2] = True

        # Update activity status of voxels accordings to new values
        self.voxel_active = ~self.voxel_empty
        # (
        #     # self.voxel_active &
        #     self.voxel_known & ~self.voxel_empty
        # )

    def increase_resolution(self):
        self.resolution = 2 * self.resolution
        shape_values = (self.resolution + 1, ) * 3

        value_known = np.full(shape_values, False)
        value_known[::2, ::2, ::2] = self.value_known
        values = upsample3d_nn(self.values)
        values = values[:-1, :-1, :-1]

        self.values = values
        self.value_known = value_known
        self.voxel_active = upsample3d_nn(self.voxel_active)

    @property
    def occupancies(self):
        return (self.values < self.threshold)

    @property
    def value_active(self):
        value_active = np.full(self.values.shape, False)
        # Active if adjacent to active voxel
        value_active[:-1, :-1, :-1] |= self.voxel_active
        value_active[:-1, :-1, 1:] |= self.voxel_active
        value_active[:-1, 1:, :-1] |= self.voxel_active
        value_active[:-1, 1:, 1:] |= self.voxel_active
        value_active[1:, :-1, :-1] |= self.voxel_active
        value_active[1:, :-1, 1:] |= self.voxel_active
        value_active[1:, 1:, :-1] |= self.voxel_active
        value_active[1:, 1:, 1:] |= self.voxel_active

        return value_active

    @property
    def voxel_known(self):
        value_known = self.value_known
        voxel_known = check_voxel_occupied(value_known)
        return voxel_known

    @property
    def voxel_empty(self):
        occ = self.occupancies
        return ~check_voxel_boundary(occ)


