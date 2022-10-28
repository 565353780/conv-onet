#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from src.common import coord2index

from conv_onet.Data.field.field import Field


class PatchPointCloudField(Field):
    ''' Patch point cloud field.

    It provides the field used for patched point cloud data. These are the points
    randomly sampled on the mesh and then partitioned.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        multi_files (callable): number of files
    '''

    def __init__(self,
                 file_name,
                 transform=None,
                 transform_add_noise=None,
                 multi_files=None):
        self.file_name = file_name
        self.transform = transform
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

        for key in vol['plane_type']:
            index[key] = coord2index(points.copy(),
                                     vol['input_vol'],
                                     reso=vol['reso'],
                                     plane=key)
            if key == 'grid':
                index[key][:, mask] = vol['reso']**3
            else:
                index[key][:, mask] = vol['reso']**2
        data['ind'] = index

        return data

    def check_complete(self, files):
        ''' Check if field is complete.

        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete
