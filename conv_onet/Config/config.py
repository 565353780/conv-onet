#!/usr/bin/env python
# -*- coding: utf-8 -*-

DATA = {
    'dataset': 'Shapes3D',
    #  'path': '/home/chli/chLi/conv-onet/demo_data/demo/Matterport3D_processed',
    'path': '/home/chli/chLi/conv-onet/demo_train',
    'classes': [''],
    'train_split': 'train',
    'val_split': 'val',
    'test_split': 'test',
    'points_subsample': 1024,
    'pointcloud_file': 'pointcloud.npz',
    'pointcloud_n': 200000,
    'pointcloud_noise': 0.0,
    'padding': 0.1,
    'unit_size': 0.004,  # define the size of a voxel, in meter
    'query_vol_size': 100,  # query crop in voxel
    #  'query_vol_size': 9,  # query crop in voxel
    'points_file': 'points_iou.npz',
    'points_iou_file': 'points_iou.npz',
}

TRAINING = {
    'batch_size': 2,
    'print_every': 100,
    'visualize_every': 10000,
    'checkpoint_every': 1000,
    'validate_every': 1000000000,  # TODO: validation for crop training
    'backup_every': 10000,
    'eval_sample': False,
    'model_selection_metric': 'iou',  # iou or loss
    'model_selection_mode': 'maximize',  # maximize or minimize
    'n_workers': 8,
    'n_workers_val': 4,
}

TEST = {
    'eval_mesh': True,
    'eval_pointcloud': False,
    'remove_wall': False,
    'model_file': './room_grid64.pt',
}

GENERATION = {
    'batch_size': 100000,
    'latent_number': 4,
    'latent_H': 8,
    'latent_W': 8,
    'latent_ny': 2,
    'latent_nx': 2,
    'latent_repeat': True,
}

CONFIG = {
    'data': DATA,
    'training': TRAINING,
    'test': TEST,
    'generation': GENERATION,
}
