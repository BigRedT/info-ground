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
from .dataset import ClusterPredDataset
from detector.model import create_detector


class MILLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_softmax = nn.LogSoftmax()
    
    def forward(self, logits, label_vec):
        log_softmax = self.log_softmax(logits) # BxC
        max_log_softmax = torch.max(log_softmax,0)[0] #C
        loss = -torch.sum(max_log_softmax*label_vec) / \
            (label_vec.sum() + 1e-6)
        
        return loss


class BatchMILLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mil_criterion = MILLoss()
    
    def forward(self, list_logits, list_label_vec):
        loss = 0
        for logits, label_vec in zip(list_logits, list_label_vec):
            loss = loss + self.mil_criterion(logits,label_vec)

        loss = loss / len(list_logits)
        return loss


def train_model(model,dataloaders,exp_const,tb_writer):
    params = [
        {'params': model.detector.parameters()},
    ]
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

    best_val_loss = 10000
    for epoch in range(exp_const.num_epochs):
        for it,data in enumerate(dataloaders['train']):
            if data is None:
                continue
            
            # Set mode
            model.detector.train()

            images = [img.cuda() for img in data['image']]
            label_vec = [vec.cuda() for vec in data['label_vec']]
            boxes = [b.cuda() for b in data['boxes']]

            logits,features = model.detector(images,boxes)
            mil_loss = model.mil_criterion(logits,label_vec)
            loss = mil_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step%exp_const.log_step==0:
                log_items = {
                    'Loss/Train': loss.item(),
                    'Lr': exp_const.lr,
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
                    'detectorr': model.detector,
                }

                for name,nn_model in save_items.items():
                    model_path = os.path.join(
                        exp_const.model_dir,
                        f'{name}_{step}')
                    torch.save({
                        'state_dict': nn_model.state_dict(),
                        'step': step},
                        model_path)

            step += 1
            #import pdb; pdb.set_trace()

def main(exp_const,data_const,model_const):
    np.random.seed(exp_const.seed)
    torch.manual_seed(exp_const.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)
    io.mkdir_if_not_exists(exp_const.log_dir)
    io.mkdir_if_not_exists(exp_const.model_dir)
    io.mkdir_if_not_exists(exp_const.vis_dir)
    
    tb_writer = SummaryWriter(log_dir=exp_const.log_dir)
    
    model_num = model_const.model_num
    save_constants({
        f'exp_{model_num}': exp_const,
        f'data_train_{model_num}': data_const['train'],
        f'model_{model_num}': model_const},
        exp_const.exp_dir)

    print('Creating dataloader ...')
    dataloaders = {}
    for mode, const in data_const.items():
        dataset = ClusterPredDataset(const)
        
        if mode=='train':
            shuffle=True
            batch_size=exp_const.train_batch_size
        else:
            shuffle=True
            batch_size=exp_const.val_batch_size

        dataloaders[mode] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=exp_const.num_workers,
            collate_fn=dataset.create_collate_fn())

    print('Creating network ...')
    model = Constants()
    model.const = model_const
    model.num_classes = len(dataloaders['train'].dataset.labels)
    model.detector = create_detector(
        extractor=True,
        num_classes=model.num_classes).cuda()
    model.mil_criterion = BatchMILLoss()
    model.detector.to_file(os.path.join(exp_const.exp_dir,'detector.txt'))

    train_model(model,dataloaders,exp_const,tb_writer)