#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import torch
import numpy as np
from torch.utils.data import dataloader, Dataset

from conv_onet.Data.field.index_field import IndexField
from conv_onet.Method.field import get_data_fields, get_inputs_field

from conv_onet.Method.common import decide_total_volume_range, update_reso


def collate_remove_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return dataloader.default_collate(batch)


def worker_init_fn(worker_id):

    def set_num_threads(nt):
        try:
            import mkl
            mkl.set_num_threads(nt)
        except:
            pass
            torch.set_num_threads(1)
            os.environ['IPC_ENABLE'] = '1'
            for o in [
                    'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                    'OMP_NUM_THREADS', 'MKL_NUM_THREADS'
            ]:
                os.environ[o] = str(nt)

    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)
    return True


class Shapes3dDataset(Dataset):

    def __init__(self,
                 dataset_folder,
                 fields,
                 split=None,
                 no_except=True,
                 transform=None,
                 cfg=None):
        ''' Initialization of the the 3D shape dataset.

        Args:
            dataset_folder (str): dataset folder
            fields (dict): dictionary of fields
            split (str): which split is used
            no_except (bool): no exception
            transform (callable): transformation applied to data points
            cfg (yaml): config file
        '''
        # Attributes
        self.dataset_folder = dataset_folder
        self.fields = fields
        self.no_except = no_except
        self.transform = transform
        self.cfg = cfg

        # Read metadata file
        metadata_file = os.path.join(dataset_folder, 'metadata.yaml')

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = yaml.load(f, Loader=yaml.FullLoader)
        else:
            self.metadata = {'': {'id': '', 'name': 'n/a'}}

        # Set index
        self.metadata['']['idx'] = 0

        # Get all models
        self.models = []
        subpath = os.path.join(dataset_folder, '')
        if not os.path.isdir(subpath):
            print("[WARN][Shapes3dDataset::__init__]")
            print("\tCategory does not exist in dataset.")

        if split is None:
            self.models += [{
                'category': '',
                'model': m
            } for m in [
                d for d in os.listdir(subpath)
                if (os.path.isdir(os.path.join(subpath, d)) and d != '')
            ]]

        else:
            split_file = os.path.join(subpath, split + '.lst')
            with open(split_file, 'r') as f:
                models_c = f.read().split('\n')

            if '' in models_c:
                models_c.remove('')

            self.models += [{'category': '', 'model': m} for m in models_c]

        # precompute
        self.split = split
        # proper resolution for feature plane/volume of the ENTIRE scene
        unit_size = self.cfg['data']['unit_size']
        recep_field = 2**6
        self.depth = 4

        #! for sliding-window case, pass all points!
        # FIXME: set 100000 to 4 for debug
        self.total_input_vol, self.total_query_vol, self.total_reso = \
            decide_total_volume_range(
                4, recep_field, unit_size)  # contain the whole scene
        return

    @classmethod
    def fromConfig(cls, mode, cfg, return_idx=False):
        dataset_folder = cfg['data']['path']

        # Get split
        splits = {
            'train': cfg['data']['train_split'],
            'val': cfg['data']['val_split'],
            'test': cfg['data']['test_split'],
        }

        split = splits[mode]

        fields = get_data_fields(mode, cfg)
        inputs_field = get_inputs_field(
            cfg['data']['pointcloud_n'],
            cfg['data']['pointcloud_noise'],
            cfg['data']['pointcloud_file'],
        )

        assert inputs_field is not None
        fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = IndexField()

        return cls(dataset_folder,
                   fields,
                   split=split,
                   cfg=cfg)

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.models)

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        category = self.models[idx]['category']
        model = self.models[idx]['model']

        model_path = os.path.join(self.dataset_folder, category, model)
        data = {}

        info = self.get_vol_info(model_path)
        data['pointcloud_crop'] = True

        for field_name, field in self.fields.items():
            try:
                field_data = field.load(model_path, idx, info)
            except Exception:
                if self.no_except:
                    print("[WARN][Shapes3dDataset::__getitem__]")
                    print("\tError occured when loading field " + field_name +
                          " of model " + model)
                    return None
                else:
                    raise

            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if k is None:
                        data[field_name] = v
                    else:
                        data['%s.%s' % (field_name, k)] = v
            else:
                data[field_name] = field_data

        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_vol_info(self, model_path):
        ''' Get crop information

        Args:
            model_path (str): path to the current data
        '''
        query_vol_size = self.cfg['data']['query_vol_size']
        unit_size = self.cfg['data']['unit_size']
        field_name = self.cfg['data']['pointcloud_file']
        recep_field = 2**6

        file_path = os.path.join(model_path, field_name)

        points_dict = np.load(file_path)
        p = points_dict['points']
        if self.split == 'train':
            # randomly sample a point as the center of input/query volume
            p_c = [
                np.random.uniform(p[:, i].min(), p[:, i].max())
                for i in range(3)
            ]
            # p_c = [np.random.uniform(-0.55, 0.55) for i in range(3)]
            p_c = np.array(p_c).astype(np.float32)

            reso = query_vol_size + recep_field - 1
            reso = update_reso(reso)
            input_vol_metric = reso * unit_size
            query_vol_metric = query_vol_size * unit_size

            # bound for the volumes
            lb_input_vol, ub_input_vol = p_c - input_vol_metric / 2, p_c + input_vol_metric / 2
            lb_query_vol, ub_query_vol = p_c - query_vol_metric / 2, p_c + query_vol_metric / 2

            input_vol = [lb_input_vol, ub_input_vol]
            query_vol = [lb_query_vol, ub_query_vol]
        else:
            reso = self.total_reso
            input_vol = self.total_input_vol
            query_vol = self.total_query_vol

        vol_info = {
            'reso': reso,
            'input_vol': input_vol,
            'query_vol': query_vol
        }
        return vol_info

    def get_model_dict(self, idx):
        return self.models[idx]

    def test_model_complete(self, category, model):
        ''' Tests if model is complete.

        Args:
            model (str): modelname
        '''
        model_path = os.path.join(self.dataset_folder, category, model)
        files = os.listdir(model_path)
        for field_name, field in self.fields.items():
            if not field.check_complete(files):
                print("[WARN][Shapes3dDataset::test_model_complete]")
                print("\tField \"" + field_name + "\" is incomplete: " +
                      model_path)
                return False

        return True
