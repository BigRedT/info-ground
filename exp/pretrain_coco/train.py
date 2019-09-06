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
from utils.constants import save_constants
from .models.NET_FILE import NET
from .dataset import DATASET


def train_model(model,dataloaders,exp_const,tb_writer):
    params = model.net.parameters()
    
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

    criterion = nn.CrossEntropyLoss()

    if model.const.model_num==-1:
        step = 0
    else:
        step = model.const.model_num

    for epoch in range(exp_const.num_epochs):
        for it,data in enumerate(dataloaders['training']):
            # Set mode
            model.net.train()

            # Forward pass
            feats = data['feats'].cuda()
            labels = data['labels'].cuda()
            logits = model.net(feats)

            # Computer loss
            loss = criterion(logits,labels)
            
            # Backward pass
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step%exp_const.log_step==0:
                log_items = {
                    'Loss': loss.data[0],
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
                    'net': model.net,
                }

                for name,nn_model in save_items.items():
                    model_path = os.path.join(
                        exp_const.model_dir,
                        f'{name}_{step}')
                    torch.save(nn_model.state_dict(),model_path)

            if step%exp_const.val_step==0:
                eval_results = eval_model(
                    model,
                    dataloaders['validation'],
                    exp_const,
                    step)
                print(eval_results)

            step += 1


def eval_model(model,dataloader,exp_const,step):
    # Set mode
    model.net.eval()

    avg_loss = 0
    num_samples = 0
    for it,data in enumerate(tqdm(dataloader)):
        if (exp_const.num_val_samples is not None) and \
            (num_samples >= exp_const.num_val_samples):
                break

        # Forward pass
        feats = data['feats'].cuda()
        labels = data['labels'].cuda()
        logits = model.net(feats)

        # Computer loss
        loss = criterion(logits,labels)    

        # Aggregate loss or accuracy
        batch_size = feats.size(0)
        num_samples += batch_size
        avg_loss += (loss.data[0]*batch_size)

    avg_loss = avg_loss / num_samples

    eval_results = {
        'Avg Loss': avg_loss, 
    }

    return eval_results


def main(exp_const,data_const,model_const):
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)
    io.mkdir_if_not_exists(exp_const.log_dir)
    io.mkdir_if_not_exists(exp_const.model_dir)
    io.mkdir_if_not_exists(exp_const.vis_dir)
    
    tb_writer = SummaryWriter()
    
    model_num = model_const.model_num
    save_constants({
        f'exp_{model_num}': exp_const,
        f'data_{model_num}': data_const,
        f'model_{model_num}': model_const},
        exp_const.exp_dir)
    
    print('Creating network ...')
    model = Model()
    model.const = model_const
    model.net = NET(model.const.net)
    if model.const.model_num != -1:
        model.net.load_state_dict(torch.load(model.const.net_path))
    model.net.cuda()
    model.img_mean = np.array([0.485, 0.456, 0.406])
    model.img_std = np.array([0.229, 0.224, 0.225])
    model.to_file(os.path.join(exp_const.exp_dir,'model.txt'))

    print('Creating dataloader ...')
    dataloaders = {}
    for mode, subset in exp_const.subset.items():
        data_const = copy.deepcopy(data_const)
        data_const.subset = subset
        dataset = DATASET(data_const)
        dataloaders[mode] = DataLoader(
            dataset,
            batch_size=exp_const.batch_size,
            shuffle=True,
            num_workers=exp_const.num_workers)

    train_model(model,dataloaders,exp_const,tb_writer)