#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch

from conv_onet.Config.config import CONFIG

from conv_onet.Data.voxel_grid import VoxelGrid
from conv_onet.Data.checkpoint_io import CheckpointIO

from conv_onet.Model.conv_onet import ConvolutionalOccupancyNetwork

from conv_onet.Dataset.shapes3d_dataset import Shapes3dDataset

from conv_onet.Method.io import export_pointcloud

from conv_onet.Module.generator3d import Generator3D


class Detector(object):

    def __init__(self):
        self.device = torch.device("cuda")
        self.out_dir = "./output/"
        self.generation_dir = self.out_dir + "generation/"

        self.cfg = CONFIG

        self.input_type = self.cfg['data']['input_type']

        self.dataset = Shapes3dDataset.fromConfig('test', self.cfg, True)

        self.model = ConvolutionalOccupancyNetwork.fromConfig(
            self.cfg, self.device, self.dataset)
        self.model.eval()

        self.checkpoint_io = CheckpointIO(self.out_dir, model=self.model)
        self.checkpoint_io.load(self.cfg['test']['model_file'])

        self.generator = Generator3D.fromConfig(self.model,
                                                self.cfg,
                                                device=self.device)

        self.test_loader = torch.utils.data.DataLoader(self.dataset,
                                                       batch_size=1,
                                                       num_workers=0,
                                                       shuffle=False)

        self.time_dicts = []
        return

    def detect(self, data):
        in_dir = self.generation_dir + 'input/'
        mesh_dir = self.generation_dir + 'meshes/'

        modelname = data['model']

        os.makedirs(mesh_dir, exist_ok=True)
        os.makedirs(in_dir, exist_ok=True)

        # Generate outputs
        out_file_dict = {}

        mesh, stats_dict = self.generator.generate_mesh_sliding(data)

        mesh_out_file = os.path.join(mesh_dir, '%s.off' % modelname)
        mesh.export(mesh_out_file)
        out_file_dict['mesh'] = mesh_out_file

        inputs_path = os.path.join(in_dir, '%s.ply' % modelname)
        inputs = data['inputs'].squeeze(0).cpu().numpy()
        export_pointcloud(inputs, inputs_path, False)
        out_file_dict['in'] = inputs_path
        return True

    def detectAll(self):
        for i, data in enumerate(self.test_loader):
            idx = data['idx'].item()
            model_dict = self.dataset.get_model_dict(idx)
            data.update(model_dict)

            print("[INFO][Detector::detectAll]")
            print("\t start detect data " + str(i) + "...")
            self.detect(data)
            break
        return True
