#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import time
import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict

from src import config
from src.checkpoints import CheckpointIO
from src.utils.io import export_pointcloud
from src.utils.voxels import VoxelGrid

from conv_onet.Model.decoder.local_decoder import LocalDecoder
from conv_onet.Model.decoder.patch_local_decoder import PatchLocalDecoder
from conv_onet.Model.decoder.local_point_decoder import LocalPointDecoder

from conv_onet.Dataset.shapes3d_dataset import Shapes3dDataset

from conv_onet.Model.encoder.pointnet.local_pool_pointnet import LocalPoolPointnet
from conv_onet.Model.encoder.pointnet.patch_local_pool_pointnet import PatchLocalPoolPointnet
from conv_onet.Model.encoder.pointnetpp.pointnet_plus_plus import PointNetPlusPlus
from conv_onet.Model.encoder.voxel.local_voxel_encoder import LocalVoxelEncoder

from conv_onet.Model.conv_onet import ConvolutionalOccupancyNetwork

from conv_onet.Method.common import update_reso

encoder_dict = {
    'pointnet_local_pool': LocalPoolPointnet,
    'pointnet_crop_local_pool': PatchLocalPoolPointnet,
    'pointnet_plus_plus': PointNetPlusPlus,
    'voxel_simple_local': LocalVoxelEncoder,
}

# Decoder dictionary
decoder_dict = {
    'simple_local': LocalDecoder,
    'simple_local_crop': PatchLocalDecoder,
    'simple_local_point': LocalPointDecoder
}


def get_model(cfg, device=None, dataset=None, **kwargs):
    decoder = cfg['model']['decoder']
    encoder = cfg['model']['encoder']
    dim = cfg['data']['dim']
    c_dim = cfg['model']['c_dim']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    padding = cfg['data']['padding']

    # for pointcloud_crop
    try:
        encoder_kwargs['unit_size'] = cfg['data']['unit_size']
        decoder_kwargs['unit_size'] = cfg['data']['unit_size']
    except:
        pass
    # local positional encoding
    if 'local_coord' in cfg['model'].keys():
        encoder_kwargs['local_coord'] = cfg['model']['local_coord']
        decoder_kwargs['local_coord'] = cfg['model']['local_coord']
    if 'pos_encoding' in cfg['model']:
        encoder_kwargs['pos_encoding'] = cfg['model']['pos_encoding']
        decoder_kwargs['pos_encoding'] = cfg['model']['pos_encoding']

    # update the feature volume/plane resolution
    if cfg['data']['input_type'] == 'pointcloud_crop':
        fea_type = cfg['model']['encoder_kwargs']['plane_type']
        if (dataset.split == 'train') or (cfg['generation']['sliding_window']):
            recep_field = 2**(
                cfg['model']['encoder_kwargs']['unet3d_kwargs']['num_levels'] +
                2)
            reso = cfg['data']['query_vol_size'] + recep_field - 1
            if 'grid' in fea_type:
                encoder_kwargs['grid_resolution'] = update_reso(
                    reso, dataset.depth)
            if bool(set(fea_type) & set(['xz', 'xy', 'yz'])):
                encoder_kwargs['plane_resolution'] = update_reso(
                    reso, dataset.depth)
        # if dataset.split == 'val': #TODO run validation in room level during training
        else:
            if 'grid' in fea_type:
                encoder_kwargs['grid_resolution'] = dataset.total_reso
            if bool(set(fea_type) & set(['xz', 'xy', 'yz'])):
                encoder_kwargs['plane_resolution'] = dataset.total_reso

    decoder = decoder_dict[decoder](dim=dim,
                                    c_dim=c_dim,
                                    padding=padding,
                                    **decoder_kwargs)

    if encoder == 'idx':
        encoder = nn.Embedding(len(dataset), c_dim)
    elif encoder is not None:
        encoder = encoder_dict[encoder](dim=dim,
                                        c_dim=c_dim,
                                        padding=padding,
                                        **encoder_kwargs)
    else:
        encoder = None

    model = ConvolutionalOccupancyNetwork(decoder, encoder, device=device)

    return model


def get_dataset(mode, cfg, return_idx=False):
    method = cfg['method']
    dataset_type = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']
    categories = cfg['data']['classes']

    # Get split
    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
    }

    split = splits[mode]

    # Create dataset
    if dataset_type == 'Shapes3D':
        # Dataset fields
        # Method specific fields (usually correspond to output)
        fields = method_dict[method].config.get_data_fields(mode, cfg)
        # Input fields
        inputs_field = get_inputs_field(mode, cfg)
        if inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = data.IndexField()

        dataset = data.Shapes3dDataset(dataset_folder,
                                       fields,
                                       split=split,
                                       categories=categories,
                                       cfg=cfg)
    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])
    return dataset


class Detector(object):

    def __init__(self):
        return

    def detect(self):
        test_config = "conv_onet/Config/pointcloud_crop/demo_matterport.yaml"

        cfg = config.load_config(test_config, 'conv_onet/Config/default.yaml')
        device = torch.device("cuda")

        out_dir = cfg['training']['out_dir']
        generation_dir = os.path.join(out_dir,
                                      cfg['generation']['generation_dir'])
        out_time_file = os.path.join(generation_dir,
                                     'time_generation_full.pkl')
        out_time_file_class = os.path.join(generation_dir,
                                           'time_generation.pkl')

        input_type = cfg['data']['input_type']
        vis_n_outputs = cfg['generation']['vis_n_outputs']
        if vis_n_outputs is None:
            vis_n_outputs = -1

        # Dataset
        dataset = config.get_dataset('test', cfg, return_idx=True)

        # Model
        model = get_model(cfg, device=device, dataset=dataset)

        checkpoint_io = CheckpointIO(out_dir, model=model)
        checkpoint_io.load(cfg['test']['model_file'])

        # Generator
        generator = config.get_generator(model, cfg, device=device)

        # Determine what to generate
        generate_mesh = cfg['generation']['generate_mesh']
        generate_pointcloud = cfg['generation']['generate_pointcloud']

        if generate_mesh and not hasattr(generator, 'generate_mesh'):
            generate_mesh = False
            print('Warning: generator does not support mesh generation.')

        if generate_pointcloud and not hasattr(generator,
                                               'generate_pointcloud'):
            generate_pointcloud = False
            print('Warning: generator does not support pointcloud generation.')

        # Loader
        test_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=1,
                                                  num_workers=0,
                                                  shuffle=False)

        # Statistics
        time_dicts = []

        # Generate
        model.eval()

        # Count how many models already created
        model_counter = defaultdict(int)

        for it, data in enumerate(tqdm(test_loader)):
            # Output folders
            mesh_dir = os.path.join(generation_dir, 'meshes')
            pointcloud_dir = os.path.join(generation_dir, 'pointcloud')
            in_dir = os.path.join(generation_dir, 'input')
            generation_vis_dir = os.path.join(generation_dir, 'vis')

            # Get index etc.
            idx = data['idx'].item()

            try:
                model_dict = dataset.get_model_dict(idx)
            except AttributeError:
                model_dict = {'model': str(idx), 'category': 'n/a'}

            modelname = model_dict['model']
            category_id = model_dict.get('category', 'n/a')

            try:
                category_name = dataset.metadata[category_id].get(
                    'name', 'n/a')
            except AttributeError:
                category_name = 'n/a'

            if category_id != 'n/a':
                mesh_dir = os.path.join(mesh_dir, str(category_id))
                pointcloud_dir = os.path.join(pointcloud_dir, str(category_id))
                in_dir = os.path.join(in_dir, str(category_id))

                folder_name = str(category_id)
                if category_name != 'n/a':
                    folder_name = str(folder_name) + '_' + category_name.split(
                        ',')[0]

                generation_vis_dir = os.path.join(generation_vis_dir,
                                                  folder_name)

            # Create directories if necessary
            if vis_n_outputs >= 0 and not os.path.exists(generation_vis_dir):
                os.makedirs(generation_vis_dir)

            if generate_mesh and not os.path.exists(mesh_dir):
                os.makedirs(mesh_dir)

            if generate_pointcloud and not os.path.exists(pointcloud_dir):
                os.makedirs(pointcloud_dir)

            if not os.path.exists(in_dir):
                os.makedirs(in_dir)

            # Timing dict
            time_dict = {
                'idx': idx,
                'class id': category_id,
                'class name': category_name,
                'modelname': modelname,
            }
            time_dicts.append(time_dict)

            # Generate outputs
            out_file_dict = {}

            # Also copy ground truth
            if cfg['generation']['copy_groundtruth']:
                modelpath = os.path.join(dataset.dataset_folder, category_id,
                                         modelname,
                                         cfg['data']['watertight_file'])
                out_file_dict['gt'] = modelpath

            if generate_mesh:
                t0 = time.time()
                if cfg['generation']['sliding_window']:
                    if it == 0:
                        print('Process scenes in a sliding-window manner')
                    out = generator.generate_mesh_sliding(data)
                else:
                    out = generator.generate_mesh(data)
                time_dict['mesh'] = time.time() - t0

                # Get statistics
                try:
                    mesh, stats_dict = out
                except TypeError:
                    mesh, stats_dict = out, {}
                time_dict.update(stats_dict)

                # Write output
                mesh_out_file = os.path.join(mesh_dir, '%s.off' % modelname)
                mesh.export(mesh_out_file)
                out_file_dict['mesh'] = mesh_out_file

            if generate_pointcloud:
                t0 = time.time()
                pointcloud = generator.generate_pointcloud(data)
                time_dict['pcl'] = time.time() - t0
                pointcloud_out_file = os.path.join(pointcloud_dir,
                                                   '%s.ply' % modelname)
                export_pointcloud(pointcloud, pointcloud_out_file)
                out_file_dict['pointcloud'] = pointcloud_out_file

            if cfg['generation']['copy_input']:
                # Save inputs
                if input_type == 'voxels':
                    inputs_path = os.path.join(in_dir, '%s.off' % modelname)
                    inputs = data['inputs'].squeeze(0).cpu()
                    voxel_mesh = VoxelGrid(inputs).to_mesh()
                    voxel_mesh.export(inputs_path)
                    out_file_dict['in'] = inputs_path
                elif input_type == 'pointcloud_crop':
                    inputs_path = os.path.join(in_dir, '%s.ply' % modelname)
                    inputs = data['inputs'].squeeze(0).cpu().numpy()
                    export_pointcloud(inputs, inputs_path, False)
                    out_file_dict['in'] = inputs_path
                elif input_type == 'pointcloud' or 'partial_pointcloud':
                    inputs_path = os.path.join(in_dir, '%s.ply' % modelname)
                    inputs = data['inputs'].squeeze(0).cpu().numpy()
                    export_pointcloud(inputs, inputs_path, False)
                    out_file_dict['in'] = inputs_path

            # Copy to visualization directory for first vis_n_output samples
            c_it = model_counter[category_id]
            if c_it < vis_n_outputs:
                # Save output files
                img_name = '%02d.off' % c_it
                for k, filepath in out_file_dict.items():
                    ext = os.path.splitext(filepath)[1]
                    out_file = os.path.join(generation_vis_dir,
                                            '%02d_%s%s' % (c_it, k, ext))
                    shutil.copyfile(filepath, out_file)

            model_counter[category_id] += 1

        # Create pandas dataframe and save
        time_df = pd.DataFrame(time_dicts)
        time_df.set_index(['idx'], inplace=True)
        time_df.to_pickle(out_time_file)

        # Create pickle files  with main statistics
        time_df_class = time_df.groupby(by=['class name']).mean()
        time_df_class.to_pickle(out_time_file_class)

        # Print results
        time_df_class.loc['mean'] = time_df_class.mean()
        print('Timings [s]:')
        print(time_df_class)
        return True
