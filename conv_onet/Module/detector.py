#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch

from conv_onet.Data.voxel_grid import VoxelGrid
from conv_onet.Data.checkpoint_io import CheckpointIO

from conv_onet.Model.conv_onet import ConvolutionalOccupancyNetwork

from conv_onet.Dataset.shapes3d_dataset import Shapes3dDataset

from conv_onet.Method.config import load_config
from conv_onet.Method.io import export_pointcloud

from conv_onet.Module.generator3d import Generator3D


class Detector(object):

    def __init__(self):
        self.test_config = "conv_onet/Config/pointcloud_crop/demo_matterport.yaml"
        self.device = torch.device("cuda")
        self.out_dir = "./output/"
        self.generation_dir = self.out_dir + "generation/"
        self.generate_mesh = True

        self.cfg = load_config(self.test_config,
                               'conv_onet/Config/default.yaml')

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
        mesh_dir = self.generation_dir + 'meshes/'
        in_dir = self.generation_dir + 'input/'

        idx = data['idx'].item()

        model_dict = self.dataset.get_model_dict(idx)
        #  model_dict = {'model': str(idx), 'category': 'n/a'}

        modelname = model_dict['model']
        category_id = model_dict.get('category', 'n/a')

        try:
            category_name = self.dataset.metadata[category_id].get(
                'name', 'n/a')
        except AttributeError:
            category_name = 'n/a'

        if category_id != 'n/a':
            mesh_dir = os.path.join(mesh_dir, str(category_id))
            in_dir = os.path.join(in_dir, str(category_id))

            folder_name = str(category_id)
            if category_name != 'n/a':
                folder_name = str(folder_name) + '_' + category_name.split(
                    ',')[0]

        os.makedirs(mesh_dir, exist_ok=True)
        os.makedirs(in_dir, exist_ok=True)

        # Generate outputs
        out_file_dict = {}

        # Also copy ground truth
        if self.cfg['generation']['copy_groundtruth']:
            modelpath = os.path.join(self.dataset.dataset_folder, category_id,
                                     modelname,
                                     self.cfg['data']['watertight_file'])
            out_file_dict['gt'] = modelpath

        if self.generate_mesh:
            if self.cfg['generation']['sliding_window']:
                out = self.generator.generate_mesh_sliding(data)
            else:
                out = self.generator.generate_mesh(data)

            try:
                mesh, stats_dict = out
            except TypeError:
                mesh, stats_dict = out, {}

            mesh_out_file = os.path.join(mesh_dir, '%s.off' % modelname)
            mesh.export(mesh_out_file)
            out_file_dict['mesh'] = mesh_out_file

        if self.cfg['generation']['copy_input']:
            # Save inputs
            if self.input_type == 'voxels':
                inputs_path = os.path.join(in_dir, '%s.off' % modelname)
                inputs = data['inputs'].squeeze(0).cpu()
                voxel_mesh = VoxelGrid(inputs).to_mesh()
                voxel_mesh.export(inputs_path)
                out_file_dict['in'] = inputs_path
            elif self.input_type == 'pointcloud_crop':
                inputs_path = os.path.join(in_dir, '%s.ply' % modelname)
                inputs = data['inputs'].squeeze(0).cpu().numpy()
                export_pointcloud(inputs, inputs_path, False)
                out_file_dict['in'] = inputs_path
            elif self.input_type == 'pointcloud' or 'partial_pointcloud':
                inputs_path = os.path.join(in_dir, '%s.ply' % modelname)
                inputs = data['inputs'].squeeze(0).cpu().numpy()
                export_pointcloud(inputs, inputs_path, False)
                out_file_dict['in'] = inputs_path
        return True

    def detectAll(self):
        for i, data in enumerate(self.test_loader):
            print("[INFO][Detector::detectAll]")
            print("\t start detect data " + str(i) + "...")
            self.detect(data)
            break
        return True
