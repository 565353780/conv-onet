#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def getPatchArray(min_point, max_point, patch_size):
    patch_min_point_array = np.mgrid[
        min_point[0]:max_point[0]:patch_size,\
        min_point[1]:max_point[1]:patch_size,\
        min_point[2]:max_point[2]:patch_size].reshape(3, -1).T
    patch_max_point_array = patch_min_point_array + patch_size
    return patch_min_point_array, patch_max_point_array
