#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import yaml
import torch
import shutil
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from collections import defaultdict

from src.checkpoints import CheckpointIO
from src.utils.io import export_pointcloud
from src.utils.voxels import VoxelGrid

from conv_onet.Data.field.index_field import IndexField
from conv_onet.Data.field.partial_point_cloud_field import PartialPointCloudField
from conv_onet.Data.field.patch_point_cloud_field import PatchPointCloudField
from conv_onet.Data.field.patch_points_field import PatchPointsField
from conv_onet.Data.field.point_cloud_field import PointCloudField
from conv_onet.Data.field.points_field import PointsField
from conv_onet.Data.field.voxels_field import VoxelsField

from conv_onet.Data.transform.pointcloud_noise import PointcloudNoise
from conv_onet.Data.transform.subsample_pointcloud import SubsamplePointcloud
from conv_onet.Data.transform.subsample_points import SubsamplePoints

from conv_onet.Model.decoder.local_decoder import LocalDecoder
from conv_onet.Model.decoder.patch_local_decoder import PatchLocalDecoder
from conv_onet.Model.decoder.local_point_decoder import LocalPointDecoder

from conv_onet.Model.encoder.pointnet.local_pool_pointnet import LocalPoolPointnet
from conv_onet.Model.encoder.pointnet.patch_local_pool_pointnet import PatchLocalPoolPointnet
from conv_onet.Model.encoder.pointnetpp.pointnet_plus_plus import PointNetPlusPlus
from conv_onet.Model.encoder.voxel.local_voxel_encoder import LocalVoxelEncoder

from conv_onet.Model.conv_onet import ConvolutionalOccupancyNetwork

from conv_onet.Dataset.shapes3d_dataset import Shapes3dDataset

from conv_onet.Method.common import decide_total_volume_range, update_reso

from conv_onet.Module.generator3d import Generator3D

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


def load_config(path, default_path=None):
    ''' Loads config file.

    Args:
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f, Loader=yaml.FullLoader)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


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


def get_data_fields(mode, cfg):
    ''' Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    points_transform = SubsamplePoints(cfg['data']['points_subsample'])

    input_type = cfg['data']['input_type']
    fields = {}
    if cfg['data']['points_file'] is not None:
        if input_type != 'pointcloud_crop':
            fields['points'] = PointsField(
                cfg['data']['points_file'],
                points_transform,
                unpackbits=cfg['data']['points_unpackbits'],
                multi_files=cfg['data']['multi_files'])
        else:
            fields['points'] = PatchPointsField(
                cfg['data']['points_file'],
                transform=points_transform,
                unpackbits=cfg['data']['points_unpackbits'],
                multi_files=cfg['data']['multi_files'])

    if mode in ('val', 'test'):
        points_iou_file = cfg['data']['points_iou_file']
        voxels_file = cfg['data']['voxels_file']
        if points_iou_file is not None:
            if input_type == 'pointcloud_crop':
                fields['points_iou'] = PatchPointsField(
                    points_iou_file,
                    unpackbits=cfg['data']['points_unpackbits'],
                    multi_files=cfg['data']['multi_files'])
            else:
                fields['points_iou'] = PointsField(
                    points_iou_file,
                    unpackbits=cfg['data']['points_unpackbits'],
                    multi_files=cfg['data']['multi_files'])
        if voxels_file is not None:
            fields['voxels'] = VoxelsField(voxels_file)

    return fields


def get_inputs_field(mode, cfg):
    ''' Returns the inputs fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): config dictionary
    '''
    input_type = cfg['data']['input_type']

    if input_type is None:
        inputs_field = None
    elif input_type == 'pointcloud':
        transform = transforms.Compose([
            SubsamplePointcloud(cfg['data']['pointcloud_n']),
            PointcloudNoise(cfg['data']['pointcloud_noise'])
        ])
        inputs_field = PointCloudField(cfg['data']['pointcloud_file'],
                                       transform,
                                       multi_files=cfg['data']['multi_files'])
    elif input_type == 'partial_pointcloud':
        transform = transforms.Compose([
            SubsamplePointcloud(cfg['data']['pointcloud_n']),
            PointcloudNoise(cfg['data']['pointcloud_noise'])
        ])
        inputs_field = PartialPointCloudField(
            cfg['data']['pointcloud_file'],
            transform,
            multi_files=cfg['data']['multi_files'])
    elif input_type == 'pointcloud_crop':
        transform = transforms.Compose([
            SubsamplePointcloud(cfg['data']['pointcloud_n']),
            PointcloudNoise(cfg['data']['pointcloud_noise'])
        ])

        inputs_field = PatchPointCloudField(
            cfg['data']['pointcloud_file'],
            transform,
            multi_files=cfg['data']['multi_files'],
        )

    elif input_type == 'voxels':
        inputs_field = VoxelsField(cfg['data']['voxels_file'])
    elif input_type == 'idx':
        inputs_field = IndexField()
    else:
        raise ValueError('Invalid input type (%s)' % input_type)
    return inputs_field


def get_dataset(mode, cfg, return_idx=False):
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
        fields = get_data_fields(mode, cfg)
        # Input fields
        inputs_field = get_inputs_field(mode, cfg)
        if inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = IndexField()

        dataset = Shapes3dDataset(dataset_folder,
                                  fields,
                                  split=split,
                                  categories=categories,
                                  cfg=cfg)
    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])
    return dataset


def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''

    if cfg['data']['input_type'] == 'pointcloud_crop':
        # calculate the volume boundary
        query_vol_metric = cfg['data']['padding'] + 1
        unit_size = cfg['data']['unit_size']
        recep_field = 2**(
            cfg['model']['encoder_kwargs']['unet3d_kwargs']['num_levels'] + 2)
        if 'unet' in cfg['model']['encoder_kwargs']:
            depth = cfg['model']['encoder_kwargs']['unet_kwargs']['depth']
        elif 'unet3d' in cfg['model']['encoder_kwargs']:
            depth = cfg['model']['encoder_kwargs']['unet3d_kwargs'][
                'num_levels']

        vol_info = decide_total_volume_range(query_vol_metric, recep_field,
                                             unit_size, depth)

        grid_reso = cfg['data']['query_vol_size'] + recep_field - 1
        grid_reso = update_reso(grid_reso, depth)
        query_vol_size = cfg['data']['query_vol_size'] * unit_size
        input_vol_size = grid_reso * unit_size
        # only for the sliding window case
        vol_bound = None
        if cfg['generation']['sliding_window']:
            vol_bound = {
                'query_crop_size': query_vol_size,
                'input_crop_size': input_vol_size,
                'fea_type': cfg['model']['encoder_kwargs']['plane_type'],
                'reso': grid_reso
            }

    else:
        vol_bound = None
        vol_info = None

    generator = Generator3D(
        model,
        device=device,
        threshold=cfg['test']['threshold'],
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        sample=cfg['generation']['use_sampling'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        input_type=cfg['data']['input_type'],
        padding=cfg['data']['padding'],
        vol_info=vol_info,
        vol_bound=vol_bound,
    )
    return generator


class Detector(object):

    def __init__(self):
        return

    def detect(self):
        test_config = "conv_onet/Config/pointcloud_crop/demo_matterport.yaml"

        cfg = load_config(test_config, 'conv_onet/Config/default.yaml')
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
        dataset = get_dataset('test', cfg, return_idx=True)

        # Model
        model = get_model(cfg, device=device, dataset=dataset)

        checkpoint_io = CheckpointIO(out_dir, model=model)
        checkpoint_io.load(cfg['test']['model_file'])

        # Generator
        generator = get_generator(model, cfg, device=device)

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
