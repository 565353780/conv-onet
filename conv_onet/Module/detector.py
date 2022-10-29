#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from conv_onet.Config.config import CONFIG

from conv_onet.Data.checkpoint_io import CheckpointIO

from conv_onet.Model.conv_onet import ConvolutionalOccupancyNetwork

from conv_onet.Dataset.shapes3d_dataset import Shapes3dDataset

from conv_onet.Method.io import export_pointcloud
from conv_onet.Method.path import createFileFolder

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
        for i, data in enumerate(self.test_loader):
            idx = data['idx'].item()
            model_dict = self.dataset.get_model_dict(idx)
            data.update(model_dict)

            print("[INFO][Detector::detectAll]")
            print("\t start detect data " + str(i) + "...")
            self.detect(data)
            break
        return True
