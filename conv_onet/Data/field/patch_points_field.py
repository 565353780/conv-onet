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
        multi_files (callable): number of files

    '''

    def __init__(self,
                 file_name,
                 transform=None,
                 unpackbits=False,
                 multi_files=None):
        self.file_name = file_name
        self.transform = transform
        self.unpackbits = unpackbits
        self.multi_files = multi_files

    def load(self, model_path, idx, vol):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            vol (dict): precomputed volume info
        '''
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name,
                                     '%s_%02d.npz' % (self.file_name, num))

        points_dict = np.load(file_path)
        points = points_dict['points']
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)

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
        for key in vol['plane_type']:
            # projected coordinates normalized to the range of [0, 1]
            p_n[key] = normalize_coord(data[None].copy(),
                                       vol['input_vol'],
                                       plane=key)
        data['normalized'] = p_n

        return data