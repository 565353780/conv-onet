#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from conv_onet.Config.config import CONFIG

from conv_onet.Data.checkpoint_io import CheckpointIO

from conv_onet.Model.conv_onet import ConvolutionalOccupancyNetwork

from conv_onet.Method.path import createFileFolder

from conv_onet.Module.unit_generator3d import UnitGenerator3D


class Detector(object):

    def __init__(self):
        self.device = torch.device("cuda")
        out_dir = "./output/"
        self.generation_dir = out_dir + "generation/"

        self.cfg = CONFIG

        self.model = ConvolutionalOccupancyNetwork.fromConfig(
            self.cfg, self.device)
        self.model.eval()

        self.checkpoint_io = CheckpointIO(out_dir, model=self.model)

        self.unit_generator = UnitGenerator3D(
            self.model, self.cfg['data']['padding'],
            self.cfg['data']['unit_size'], self.cfg['data']['query_vol_size'])

        self.loadModel(self.cfg['test']['model_file'])
        return

    def loadModel(self, model_file_path):
        self.checkpoint_io.load(model_file_path)
        return True

    def detect(self, point_array, render=False, print_progress=False):
        result = {}

        self.unit_generator.reconSpace(point_array, render, print_progress)
        crop_space = self.unit_generator.crop_space

        result['encode'] = crop_space.getFeatureArray('encode')
        result['occ'] = crop_space.getFeatureArray('occ')
        result['mask'] = crop_space.getFeatureMaskArray('valid')
        result['mask_feature_idx'] = crop_space.getMaskFeatureIdxArray('valid')
        return result

    def detectAndSave(self, point_array, render=False, print_progress=False):
        modelname = 'test'

        save_input_file_path = self.generation_dir + 'input/' + modelname + '.ply'
        save_mesh_file_path = self.generation_dir + 'meshes/' + modelname + '.off'

        createFileFolder(save_input_file_path)
        createFileFolder(save_mesh_file_path)

        mesh, stats_dict = self.unit_generator.generate_mesh_sliding(
            point_array, render, print_progress)

        mesh.export(save_mesh_file_path)
        return True
