#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch import distributions as dist

from conv_onet.Model.decoder.local_decoder import LocalDecoder
from conv_onet.Model.decoder.patch_local_decoder import PatchLocalDecoder
from conv_onet.Model.decoder.local_point_decoder import LocalPointDecoder

from conv_onet.Model.encoder.pointnet.local_pool_pointnet import LocalPoolPointnet
from conv_onet.Model.encoder.pointnet.patch_local_pool_pointnet import PatchLocalPoolPointnet
from conv_onet.Model.encoder.pointnetpp.pointnet_plus_plus import PointNetPlusPlus
from conv_onet.Model.encoder.voxel.local_voxel_encoder import LocalVoxelEncoder

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


class ConvolutionalOccupancyNetwork(nn.Module):

    def __init__(self, decoder, encoder=None, device=None):
        super().__init__()

        self.decoder = decoder.to(device)

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device

    @classmethod
    def fromConfig(cls, cfg, device=None):
        encoder = cfg['model']['encoder']
        decoder = cfg['model']['decoder']

        dim = cfg['data']['dim']
        c_dim = cfg['model']['c_dim']

        encoder_kwargs = cfg['model']['encoder_kwargs']
        decoder_kwargs = cfg['model']['decoder_kwargs']

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
        fea_type = cfg['model']['encoder_kwargs']['plane_type']
        recep_field = 2**(
            cfg['model']['encoder_kwargs']['unet3d_kwargs']['num_levels'] + 2)
        reso = cfg['data']['query_vol_size'] + recep_field - 1

        depth = cfg['model']['encoder_kwargs']['unet3d_kwargs']['num_levels']
        if 'grid' in fea_type:
            encoder_kwargs['grid_resolution'] = update_reso(
                reso, depth)
        if bool(set(fea_type) & set(['xz', 'xy', 'yz'])):
            encoder_kwargs['plane_resolution'] = update_reso(
                reso, depth)

        decoder = decoder_dict[decoder](dim=dim,
                                        c_dim=c_dim,
                                        padding=padding,
                                        **decoder_kwargs)

        encoder = encoder_dict[encoder](dim=dim,
                                        c_dim=c_dim,
                                        padding=padding,
                                        **encoder_kwargs)
        return cls(decoder, encoder, device=device)

    def forward(self, p, inputs, sample=True, **kwargs):
        c = self.encode_inputs(inputs)
        p_r = self.decode(p, c, **kwargs)
        return p_r

    def encode_inputs(self, inputs):
        assert self.encoder is not None

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            #FIXME: Return inputs?
            c = torch.empty(inputs.size(0), 0)
        return c

    def decode(self, p, c, **kwargs):
        logits = self.decoder(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def to(self, device):
        model = super().to(device)
        model._device = device
        return model
