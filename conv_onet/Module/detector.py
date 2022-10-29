#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np

from conv_onet.Config.config import CONFIG

from conv_onet.Data.checkpoint_io import CheckpointIO

from conv_onet.Model.conv_onet import ConvolutionalOccupancyNetwork

from conv_onet.Dataset.shapes3d_dataset import Shapes3dDataset

from conv_onet.Method.common import decide_total_volume_range, update_reso
from conv_onet.Method.io import export_pointcloud
from conv_onet.Method.path import createFileFolder

from conv_onet.Module.generator3d import Generator3D


class Detector(object):

    def __init__(self):
        self.device = torch.device("cuda")
        out_dir = "./output/"
        self.generation_dir = out_dir + "generation/"

        self.cfg = CONFIG

        self.dataset = Shapes3dDataset.fromConfig('test', self.cfg, True)

        self.model = ConvolutionalOccupancyNetwork.fromConfig(
            self.cfg, self.device, self.dataset)
        self.model.eval()

        self.checkpoint_io = CheckpointIO(out_dir, model=self.model)

        self.generator = Generator3D.fromConfig(self.model,
                                                self.cfg,
                                                device=self.device)

        self.test_loader = torch.utils.data.DataLoader(self.dataset,
                                                       batch_size=1,
                                                       num_workers=0,
                                                       shuffle=False)

        self.loadModel(self.cfg['test']['model_file'])
        return

    def loadModel(self, model_file_path):
        self.checkpoint_io.load(model_file_path)
        return True

    def detect(self, data):
        modelname = data['model']

        save_input_file_path = self.generation_dir + 'input/' + modelname + '.ply'
        save_mesh_file_path = self.generation_dir + 'meshes/' + modelname + '.off'

        createFileFolder(save_input_file_path)
        createFileFolder(save_mesh_file_path)

        mesh, stats_dict = self.generator.generate_mesh_sliding(data)

        mesh.export(save_mesh_file_path)

        inputs = data['inputs'].squeeze(0).cpu().numpy()
        export_pointcloud(inputs, save_input_file_path, False)
        return True

    def detectAll(self):
        model_path = \
            "/home/chli/chLi/conv-onet/demo_data/demo/Matterport3D_processed/17DRP5sb8fy/"
        file_path = model_path + "pointcloud.npz"
        points_dict = np.load(file_path)
        p = points_dict['points']

        data = {
            'category': '',
            'model': '17DRP5sb8fy',
        }

        for i, data in enumerate(self.test_loader):
            idx = data['idx'].item()
            model_dict = self.dataset.get_model_dict(idx)
            data.update(model_dict)

            print("[INFO][Detector::detectAll]")
            print("\t start detect data " + str(i) + "...")
            self.detect(data)
            break
        return True
