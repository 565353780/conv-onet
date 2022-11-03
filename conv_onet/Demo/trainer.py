#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import datetime
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from conv_onet.Config.config import CONFIG

from conv_onet.Data.checkpoint_io import CheckpointIO

from conv_onet.Model.conv_onet import ConvolutionalOccupancyNetwork

from conv_onet.Dataset.shapes3d_dataset import Shapes3dDataset, collate_remove_none, worker_init_fn

from conv_onet.Method.time import getCurrentTime

from conv_onet.Module.generator3d import Generator3D
from conv_onet.Module.trainer import Trainer


def demo():
    cfg = CONFIG
    device = torch.device("cuda")

    batch_size = cfg['training']['batch_size']
    backup_every = cfg['training']['backup_every']

    model_selection_metric = cfg['training']['model_selection_metric']
    if cfg['training']['model_selection_mode'] == 'maximize':
        model_selection_sign = 1
    else:
        model_selection_sign = -1

    current_time_str = getCurrentTime()
    save_model_dir = "./output/models/" + current_time_str + "/"
    log_dir = "./logs/" + current_time_str + "/"
    os.makedirs(save_model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_dataset = Shapes3dDataset.fromConfig('train', cfg)
    val_dataset = Shapes3dDataset.fromConfig('val', cfg, return_idx=True)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=cfg['training']['n_workers'],
                              shuffle=True,
                              collate_fn=collate_remove_none,
                              worker_init_fn=worker_init_fn)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            num_workers=cfg['training']['n_workers_val'],
                            shuffle=False,
                            collate_fn=collate_remove_none,
                            worker_init_fn=worker_init_fn)

    model = ConvolutionalOccupancyNetwork.fromConfig(cfg, device)

    generator = Generator3D.fromConfig(model, cfg, device)

    # Intialize training
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    trainer = Trainer.fromConfig(model, optimizer, cfg, device=device)

    checkpoint_io = CheckpointIO(save_model_dir,
                                 model=model,
                                 optimizer=optimizer)
    try:
        load_dict = checkpoint_io.load('model.pt')
    except FileExistsError:
        load_dict = dict()
    epoch_it = load_dict.get('epoch_it', 0)
    it = load_dict.get('it', 0)
    metric_val_best = load_dict.get('loss_val_best',
                                    -model_selection_sign * np.inf)

    if metric_val_best == np.inf or metric_val_best == -np.inf:
        metric_val_best = -model_selection_sign * np.inf
    print('Current best validation metric (%s): %.8f' %
          (model_selection_metric, metric_val_best))
    logger = SummaryWriter(log_dir)

    # Shorthands
    print_every = cfg['training']['print_every']
    checkpoint_every = cfg['training']['checkpoint_every']
    validate_every = cfg['training']['validate_every']
    visualize_every = cfg['training']['visualize_every']

    # Print model
    nparameters = sum(p.numel() for p in model.parameters())
    print('Total number of parameters: %d' % nparameters)

    while True:
        epoch_it += 1

        for batch in tqdm(train_loader):
            it += 1
            loss = trainer.train_step(batch)
            logger.add_scalar('train/loss', loss, it)

            if print_every > 0 and (it % print_every) == 0:
                t = datetime.datetime.now()
                print('[Epoch %02d] it=%03d, loss=%.4f, %02d:%02d' %
                      (epoch_it, it, loss, t.hour, t.minute))

            if visualize_every > 0 and (it % visualize_every) == 0:
                print('Visualizing')
                # FIXME: finish it later
                if False:
                    mesh, stats_dict = generator.generate_mesh_sliding('data')
                    mesh.export("test.off")

            if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
                print('Saving checkpoint')
                checkpoint_io.save('model.pt',
                                   epoch_it=epoch_it,
                                   it=it,
                                   loss_val_best=metric_val_best)

            if (backup_every > 0 and (it % backup_every) == 0):
                print('Backup checkpoint')
                checkpoint_io.save('model_%d.pt' % it,
                                   epoch_it=epoch_it,
                                   it=it,
                                   loss_val_best=metric_val_best)

            if validate_every > 0 and (it % validate_every) == 0:
                eval_dict = trainer.evaluate(val_loader)
                metric_val = eval_dict[model_selection_metric]
                print('Validation metric (%s): %.4f' %
                      (model_selection_metric, metric_val))

                for k, v in eval_dict.items():
                    logger.add_scalar('val/%s' % k, v, it)

                if model_selection_sign * (metric_val - metric_val_best) > 0:
                    metric_val_best = metric_val
                    print('New best model (loss %.4f)' % metric_val_best)
                    checkpoint_io.save('model_best.pt',
                                       epoch_it=epoch_it,
                                       it=it,
                                       loss_val_best=metric_val_best)
    return True
