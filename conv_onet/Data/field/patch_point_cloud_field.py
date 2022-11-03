#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np

from conv_onet.Data.field.field import Field

from conv_onet.Method.common import coord2index


class PatchPointCloudField(Field):
    ''' Patch point cloud field.

    It provides the field used for patched point cloud data. These are the points
    randomly sampled on the mesh and then partitioned.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
    '''

    def __init__(self, file_name, transform=None):
        self.file_name = file_name
        self.transform = transform
        return

    def load(self, model_path, idx, vol):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            vol (dict): precomputed volume info
        '''
        file_path = os.path.join(model_path, self.file_name)

        pointcloud_dict = np.load(file_path)

        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)

        # add noise globally
        if self.transform is not None:
            data = {None: points, 'normals': normals}
            data = self.transform(data)
            points = data[None]

        # acquire the crop index
        ind_list = []
        for i in range(3):
            ind_list.append((points[:, i] >= vol['input_vol'][0][i])
                            & (points[:, i] <= vol['input_vol'][1][i]))
        mask = ind_list[0] & ind_list[1] & ind_list[
            2]  # points inside the input volume
        mask = ~mask  # True means outside the boundary!!
        data['mask'] = mask
        points[mask] = 0.0

        # calculate index of each point w.r.t. defined resolution
        index = {}

        index['grid'] = coord2index(points.copy(),
                                    vol['input_vol'],
                                    reso=vol['reso'])
        index['grid'][:, mask] = vol['reso']**3
        data['ind'] = index

        return data

    def check_complete(self, files):
        ''' Check if field is complete.

        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete
