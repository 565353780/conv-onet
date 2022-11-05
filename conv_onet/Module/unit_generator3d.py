#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import torch
import trimesh
import numpy as np
from tqdm import trange

from conv_onet.Lib.libmcubes.mcubes import marching_cubes

from conv_onet.Data.crop_space import CropSpace

from conv_onet.Config.crop import UNIT_LB, UNIT_UB

from conv_onet.Method.common import \
    normalize_coord, add_key, coord2index, decide_total_volume_range, update_reso
from conv_onet.Method.patch import getPatchArray


class UnitGenerator3D(object):

    def __init__(self,
                 model,
                 padding,
                 unit_size,
                 query_vol_size,
                 device=torch.device('cuda'),
                 threshold=0.2,
                 resolution0=128):
        self.points_batch_size = 100000

        self.model = model.to(device)
        self.threshold = threshold
        self.device = device
        self.resolution0 = resolution0
        self.padding = padding

        query_vol_metric = padding + 1

        self.input_vol, _, _ = decide_total_volume_range(
            query_vol_metric, 2**6, unit_size)

        grid_reso = query_vol_size + 2**6 - 1
        grid_reso = update_reso(grid_reso)
        query_vol_size = query_vol_size * unit_size
        input_vol_size = grid_reso * unit_size

        self.vol_bound = {
            'query_crop_size': query_vol_size,
            'input_crop_size': input_vol_size,
            'reso': grid_reso
        }

        self.setUnitCropBound()

        self.crop_space = CropSpace(self.vol_bound['input_crop_size'],
                                    self.vol_bound['query_crop_size'])
        return

    @classmethod
    def fromConfig(cls, model, cfg):
        padding = cfg['data']['padding']
        unit_size = cfg['data']['unit_size']
        query_vol_size = cfg['data']['query_vol_size']

        return cls(model, padding, unit_size, query_vol_size)

    def setUnitCropBound(self):
        query_crop_size = self.vol_bound['query_crop_size']
        input_crop_size = self.vol_bound['input_crop_size']

        self.vol_bound['axis_n_crop'] = np.ceil(
            (UNIT_UB - UNIT_LB) / query_crop_size).astype(int)

        lb_query, ub_query = getPatchArray(UNIT_LB, UNIT_UB, query_crop_size)

        center = (lb_query + ub_query) / 2
        lb_input = center - input_crop_size / 2
        ub_input = center + input_crop_size / 2

        num_crop = np.prod(self.vol_bound['axis_n_crop'])

        self.vol_bound['n_crop'] = num_crop
        self.vol_bound['input_vol'] = np.stack([lb_input, ub_input], axis=1)
        self.vol_bound['query_vol'] = np.stack([lb_query, ub_query], axis=1)
        return True

    def encode_crop(self, inputs, device, vol_bound):
        ''' Encode a crop to feature volumes

        Args:
            inputs (dict): input point cloud
            device (device): pytorch device
            vol_bound (dict): volume boundary
        '''
        index = {}
        # crop the input point cloud
        mask_x = (inputs[:, :, 0] >= vol_bound['input_vol'][0][0]) &\
                (inputs[:, :, 0] < vol_bound['input_vol'][1][0])
        mask_y = (inputs[:, :, 1] >= vol_bound['input_vol'][0][1]) &\
                (inputs[:, :, 1] < vol_bound['input_vol'][1][1])
        mask_z = (inputs[:, :, 2] >= vol_bound['input_vol'][0][2]) &\
                (inputs[:, :, 2] < vol_bound['input_vol'][1][2])
        mask = mask_x & mask_y & mask_z

        p_input = inputs[mask]
        if p_input.shape[0] == 0:  # no points in the current crop
            return None

        ind = coord2index(p_input.clone(),
                          vol_bound['input_vol'],
                          reso=self.vol_bound['reso'])
        index['grid'] = ind.unsqueeze(0)
        input_cur = add_key(p_input.unsqueeze(0),
                            index,
                            'points',
                            'index',
                            device=device)

        with torch.no_grad():
            c = self.model.encode_inputs(input_cur)
        return c

    def predict_crop_occ(self, pi, c, vol_bound=None):
        ''' Predict occupancy values for a crop

        Args:
            pi (dict): query points
            c (tensor): encoded feature volumes
            vol_bound (dict): volume boundary
        '''
        occ_hat = pi.new_empty((pi.shape[0]))

        if pi.shape[0] == 0:
            return occ_hat
        pi_in = pi.unsqueeze(0)
        pi_in = {'p': pi_in}
        p_n = {}
        # projected coordinates normalized to the range of [0, 1]
        p_n['grid'] = normalize_coord(
            pi.clone(), vol_bound['input_vol']).unsqueeze(0).to(self.device)
        pi_in['p_n'] = p_n

        # predict occupancy of the current crop
        with torch.no_grad():
            occ_cur = self.model.decode(pi_in, c).logits
        occ_hat = occ_cur.squeeze(0)
        return occ_hat

    def eval_points(self, p, c=None, vol_bound=None):
        ''' Evaluates the occupancy values for the points.

        Args:
            p (tensor): points
            c (tensor): encoded feature volumes
        '''
        if c is None:
            return None

        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []
        for pi in p_split:
            occ_hat = self.predict_crop_occ(pi, c, vol_bound=vol_bound)
            occ_hats.append(occ_hat)

        occ_hat = torch.cat(occ_hats, dim=0)
        return occ_hat

    def extract_mesh(self, occ_hat, stats_dict=dict()):
        ''' Extracts the mesh from the predicted occupancy grid.

        Args:
            occ_hat (tensor): value grid of occupancies
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)

        # Make sure that mesh is watertight
        t0 = time.time()
        occ_hat_padded = np.pad(occ_hat, 1, 'constant', constant_values=-1e6)
        vertices, triangles = marching_cubes(occ_hat_padded, threshold)
        stats_dict['time (marching cubes)'] = time.time() - t0
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # # Undo padding
        vertices -= 1

        # Scale the mesh back to its original metric
        bb_min = self.vol_bound['query_vol'][:, 0].min(axis=0)
        bb_max = self.vol_bound['query_vol'][:, 1].max(axis=0)
        mc_unit = max(bb_max - bb_min) / (self.vol_bound['axis_n_crop'].max() *
                                          self.resolution0)
        vertices = vertices * mc_unit + bb_min

        # Create mesh
        mesh = trimesh.Trimesh(vertices,
                               triangles,
                               vertex_normals=None,
                               process=False)

        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh
        return mesh

    def generate_mesh_sliding(self, data):
        ''' Generates the output mesh in sliding-window manner.
            Adapt for real-world scale.

        Args:
            data (tensor): data tensor
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}

        inputs = data.get('inputs', torch.empty(1, 0)).to(device)

        self.crop_space.updatePointArray(inputs)
        exit()

        # acquire the boundary for every crops
        #  self.get_crop_bound(inputs)

        nx = self.resolution0
        n_crop = self.vol_bound['n_crop']
        n_crop_axis = self.vol_bound['axis_n_crop']

        # occupancy in each direction
        r = nx
        occ_values = np.array([]).reshape(r, r, 0)
        occ_values_y = np.array([]).reshape(r, 0, r * n_crop_axis[2])
        occ_values_x = np.array([]).reshape(0, r * n_crop_axis[1],
                                            r * n_crop_axis[2])
        for i in trange(n_crop):
            # encode the current crop
            vol_bound = {}
            vol_bound['query_vol'] = self.vol_bound['query_vol'][i]
            vol_bound['input_vol'] = self.vol_bound['input_vol'][i]
            c = self.encode_crop(inputs, device, vol_bound)

            if c is not None:
                bb_min = self.vol_bound['query_vol'][i][0]
                bb_max = bb_min + self.vol_bound['query_crop_size']

                t = (bb_max - bb_min) / nx  # inteval
                pp = np.mgrid[bb_min[0]:bb_max[0]:t[0],
                              bb_min[1]:bb_max[1]:t[1],
                              bb_min[2]:bb_max[2]:t[2]].reshape(3, -1).T
                pp = torch.from_numpy(pp).to(device)
                values = self.eval_points(
                    pp, c, vol_bound=vol_bound).detach().cpu().numpy()
                values = values.reshape(nx, nx, nx)
            else:
                values = np.ones([nx, nx, nx]) * -1e6

            # concatenate occ_value along every axis
            # along z axis
            occ_values = np.concatenate((occ_values, values), axis=2)
            # along y axis
            if (i + 1) % n_crop_axis[2] == 0:
                occ_values_y = np.concatenate((occ_values_y, occ_values),
                                              axis=1)
                occ_values = np.array([]).reshape(r, r, 0)
            # along x axis
            if (i + 1) % (n_crop_axis[2] * n_crop_axis[1]) == 0:
                occ_values_x = np.concatenate((occ_values_x, occ_values_y),
                                              axis=0)
                occ_values_y = np.array([]).reshape(r, 0, r * n_crop_axis[2])

        value_grid = occ_values_x
        mesh = self.extract_mesh(value_grid, stats_dict=stats_dict)
        return mesh, stats_dict
