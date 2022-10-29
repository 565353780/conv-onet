#!/usr/bin/env python
# -*- coding: utf-8 -*-

DATA = {
    'dataset': 'Shapes3D',
    'path': '/home/chli/chLi/conv-onet/demo_data/demo/Matterport3D_processed',
    'classes': [''],
    'train_split': 'train',
    'val_split': 'val',
    'test_split': 'test',
    'dim': 3,
    'points_subsample': 1024,
    'pointcloud_file': 'pointcloud.npz',
    'pointcloud_n': 200000,
    'pointcloud_noise': 0.0,
    'padding': 0.1,
    'unit_size': 0.02,  # define the size of a voxel, in meter
    'query_vol_size': 90,  # query crop in voxel
}

MODEL = {
    'local_coord': True,
    'decoder': 'simple_local_crop',
    'encoder': 'pointnet_crop_local_pool',
    'decoder_kwargs': {
        'sample_mode': 'bilinear',  # bilinear / nearest
        'hidden_size': 32,
    },
    'encoder_kwargs': {
        'hidden_dim': 32,
        'plane_type': ['grid'],
        'unet3d': True,
        'unet3d_kwargs': {
            'num_levels': 4,  # define the receptive field, 3 -> 32, 4 -> 64
            'f_maps': 32,
            'in_channels': 32,
            'out_channels': 32,
        },
    },
    'c_dim': 32,
}

TRAINING = {
    'out_dir': 'out/pointcloud_crop_training',
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
    'threshold': 0.2,
    'eval_mesh': True,
    'eval_pointcloud': False,
    'remove_wall': False,
    'model_file': './room_grid64.pt'
}

GENERATION = {
    'batch_size': 100000,
    'refinement_step': 0,
    'use_sampling': False,
    'resolution_0': 128,
    'upsampling_steps': 0,
    'simplify_nfaces': None,
    'latent_number': 4,
    'latent_H': 8,
    'latent_W': 8,
    'latent_ny': 2,
    'latent_nx': 2,
    'latent_repeat': True,
}

CONFIG = {
    'data': DATA,
    'model': MODEL,
    'training': TRAINING,
    'test': TEST,
    'generation': GENERATION,
}
