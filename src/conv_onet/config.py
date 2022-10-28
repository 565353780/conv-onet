import torch
import torch.distributions as dist
from torch import nn
import os
from src.encoder import encoder_dict
from src.conv_onet import models, training
from src.conv_onet import generation
from src import data
from src import config
from src.common import decide_total_volume_range, update_reso
from torchvision import transforms
import numpy as np


def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    '''
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
            recep_field = 2**(cfg['model']['encoder_kwargs']['unet3d_kwargs']['num_levels'] + 2)
            reso = cfg['data']['query_vol_size'] + recep_field - 1
            if 'grid' in fea_type:
                encoder_kwargs['grid_resolution'] = update_reso(reso, dataset.depth)
            if bool(set(fea_type) & set(['xz', 'xy', 'yz'])):
                encoder_kwargs['plane_resolution'] = update_reso(reso, dataset.depth)
        # if dataset.split == 'val': #TODO run validation in room level during training
        else:
            if 'grid' in fea_type:
                encoder_kwargs['grid_resolution'] = dataset.total_reso
            if bool(set(fea_type) & set(['xz', 'xy', 'yz'])):
                encoder_kwargs['plane_resolution'] = dataset.total_reso
    

    decoder = models.decoder_dict[decoder](
        dim=dim, c_dim=c_dim, padding=padding,
        **decoder_kwargs
    )

    if encoder == 'idx':
        encoder = nn.Embedding(len(dataset), c_dim)
    elif encoder is not None:
        encoder = encoder_dict[encoder](
            dim=dim, c_dim=c_dim, padding=padding,
            **encoder_kwargs
        )
    else:
        encoder = None

    model = models.ConvolutionalOccupancyNetwork(
        decoder, encoder, device=device
    )

    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    input_type = cfg['data']['input_type']

    trainer = training.Trainer(
        model, optimizer,
        device=device, input_type=input_type,
        vis_dir=vis_dir, threshold=threshold,
        eval_sample=cfg['training']['eval_sample'],
    )

    return trainer



def get_data_fields(mode, cfg):
    ''' Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    points_transform = data.SubsamplePoints(cfg['data']['points_subsample'])
    
    input_type = cfg['data']['input_type']
    fields = {}
    if cfg['data']['points_file'] is not None:
        if input_type != 'pointcloud_crop':
            fields['points'] = data.PointsField(
                cfg['data']['points_file'], points_transform,
                unpackbits=cfg['data']['points_unpackbits'],
                multi_files=cfg['data']['multi_files']
            )
        else:
            fields['points'] = data.PatchPointsField(
                cfg['data']['points_file'], 
                transform=points_transform,
                unpackbits=cfg['data']['points_unpackbits'],
                multi_files=cfg['data']['multi_files']
            )

    
    if mode in ('val', 'test'):
        points_iou_file = cfg['data']['points_iou_file']
        voxels_file = cfg['data']['voxels_file']
        if points_iou_file is not None:
            if input_type == 'pointcloud_crop':
                fields['points_iou'] = data.PatchPointsField(
                points_iou_file,
                unpackbits=cfg['data']['points_unpackbits'],
                multi_files=cfg['data']['multi_files']
                )
            else:
                fields['points_iou'] = data.PointsField(
                    points_iou_file,
                    unpackbits=cfg['data']['points_unpackbits'],
                    multi_files=cfg['data']['multi_files']
                )
        if voxels_file is not None:
            fields['voxels'] = data.VoxelsField(voxels_file)

    return fields
