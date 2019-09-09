import os
import h5py
import math
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import itertools
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import utils.io as io
from utils.constants import save_constants, Constants
from .object_encoder import ObjectEncoder
from .dataset import DetFeatDataset
from .info_nce_loss import InfoNCE


def create_info_nce_criterion(x_dim,c_dim,d):
    fx = nn.Sequential(
        nn.Linear(x_dim,d),
        nn.ReLU(),
        nn.Linear(d,d))

    fy = nn.Sequential(
        nn.Linear(c_dim,d),
        nn.ReLU(),
        nn.Linear(d,d))

    criterion = InfoNCE(fx.cuda(),fy.cuda())
    
    return criterion


def train_model(model,dataloaders,exp_const,tb_writer):
    criterion = create_info_nce_criterion(
        model.object_encoder.const.object_feature_dim,
        model.object_encoder.const.context_layer.hidden_size,
        model.object_encoder.const.context_layer.hidden_size//2)
    
    params = itertools.chain(
        model.object_encoder.parameters(),
        criterion.parameters())
    
    if exp_const.optimizer == 'SGD':
        opt = optim.SGD(
            params,
            lr=exp_const.lr,
            momentum=exp_const.momentum)
    elif exp_const.optimizer == 'Adam':
        opt = optim.Adam(
            params,
            lr=exp_const.lr)
    else:
        assert(False), 'optimizer not implemented'

    if model.const.model_num==-1:
        step = 0
    else:
        step = model.const.model_num
        criterion.load_state_dict(torch.load(
            os.path.join(
                exp_const.model_dir,
                f'self_sup_criterion_{step}')))


    for epoch in range(exp_const.num_epochs):
        for it,data in enumerate(dataloaders['train']):
            # Set mode
            model.object_encoder.train()
            criterion.train()

            # Forward pass
            object_features = data['features'].cuda()
            object_mask = data['object_mask'].cuda()
            pad_mask = data['pad_mask'].cuda()
            context_object_features = model.object_encoder(
                object_features,
                object_mask,
                pad_mask)
                
            # Computer loss
            loss = criterion(
                object_features,
                context_object_features,
                object_mask)
            
            # Backward pass
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step%exp_const.log_step==0:
                log_items = {
                    'Loss': loss.item(),
                }

                log_str = f'Epoch: {epoch} | Iter: {it} | Step: {step} | '
                for name,value in log_items.items():
                    log_str += '{}: {:.4f} | '.format(name,value)
                    tb_writer.add_scalar(name,value,step)

                print(log_str)
            
            if step%(50*exp_const.log_step)==0:
                print(f'Experiment: {exp_const.exp_name}')
                
            if step%exp_const.model_save_step==0:
                save_items = {
                    'object_encoder': model.object_encoder,
                    'self_sup_criterion': criterion,
                }

                for name,nn_model in save_items.items():
                    model_path = os.path.join(
                        exp_const.model_dir,
                        f'{name}_{step}')
                    torch.save(nn_model.state_dict(),model_path)

            # if step%exp_const.val_step==0:
            #     eval_results = eval_model(
            #         model,
            #         dataloaders['val'],
            #         exp_const,
            #         step)
            #     print(eval_results)

            step += 1


# def eval_model(model,dataloader,exp_const,step):
#     # Set mode
#     model.net.eval()

#     avg_loss = 0
#     num_samples = 0
#     for it,data in enumerate(tqdm(dataloader)):
#         if (exp_const.num_val_samples is not None) and \
#             (num_samples >= exp_const.num_val_samples):
#                 break

#         # Forward pass
#         feats = data['feats'].cuda()
#         labels = data['labels'].cuda()
#         logits = model.net(feats)

#         # Computer loss
#         loss = criterion(logits,labels)    

#         # Aggregate loss or accuracy
#         batch_size = feats.size(0)
#         num_samples += batch_size
#         avg_loss += (loss.data[0]*batch_size)

#     avg_loss = avg_loss / num_samples

#     eval_results = {
#         'Avg Loss': avg_loss, 
#     }

#     return eval_results


def main(exp_const,data_const,model_const):
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)
    io.mkdir_if_not_exists(exp_const.log_dir)
    io.mkdir_if_not_exists(exp_const.model_dir)
    io.mkdir_if_not_exists(exp_const.vis_dir)
    
    tb_writer = SummaryWriter(log_dir=exp_const.log_dir)
    
    model_num = model_const.model_num
    save_constants({
        f'exp_{model_num}': exp_const,
        f'data_train_{model_num}': data_const['train'],
        f'data_val_{model_num}': data_const['val'],
        f'model_{model_num}': model_const},
        exp_const.exp_dir)
    
    print('Creating network ...')
    model = Constants()
    model.const = model_const
    model.object_encoder = ObjectEncoder(model.const.object_encoder)
    if model.const.model_num != -1:
        model.object_encoder.load_state_dict(
            torch.load(model.const.object_encoder_path))
    model.object_encoder.cuda()
    model.object_encoder.to_file(
        os.path.join(exp_const.exp_dir,'object_encoder.txt'))

    print('Creating dataloader ...')
    dataloaders = {}
    for mode, const in data_const.items():
        dataset = DetFeatDataset(const)
        
        if mode=='train':
            shuffle=True
        else:
            shuffle=False

        dataloaders[mode] = DataLoader(
            dataset,
            batch_size=exp_const.batch_size,
            shuffle=shuffle,
            num_workers=exp_const.num_workers)

    train_model(model,dataloaders,exp_const,tb_writer)