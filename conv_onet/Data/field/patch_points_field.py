#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np

from conv_onet.Data.field.field import Field

from conv_onet.Method.common import normalize_coord


class PatchPointsField(Field):
    ''' Patch Point Field.

    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape and then split to patches.

    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the points tensor
    '''

    def __init__(self, file_name, transform=None, unpackbits=False):
        self.file_name = file_name
        self.transform = transform
        self.unpackbits = unpackbits

    def load(self, model_path, idx, vol):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            vol (dict): precomputed volume info
        '''
        file_path = os.path.join(model_path, self.file_name)

        points_dict = np.load(file_path)
        points = points_dict['points']
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)

        if 'occupancies' not in points_dict.keys():
            print("[ERROR][PatchPointsField::load]")
            print("\t occupancies not in points_dict.keys()!")
            return None
        occupancies = points_dict['occupancies']
        if self.unpackbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]
        occupancies = occupancies.astype(np.float32)

        # acquire the crop
        ind_list = []
        for i in range(3):
            ind_list.append((points[:, i] >= vol['query_vol'][0][i])
                            & (points[:, i] <= vol['query_vol'][1][i]))
        ind = ind_list[0] & ind_list[1] & ind_list[2]
        data = {
            None: points[ind],
            'occ': occupancies[ind],
        }

        if self.transform is not None:
            data = self.transform(data)

        # calculate normalized coordinate w.r.t. defined query volume
        p_n = {}
        # projected coordinates normalized to the range of [0, 1]
        p_n['grid'] = normalize_coord(data[None].copy(),
                                      vol['input_vol'],
                                      plane='grid')
        data['normalized'] = p_n

        return data
