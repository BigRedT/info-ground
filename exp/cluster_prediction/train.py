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
        self.softmax_1 = nn.Softmax(1)
        self.softmax_0 = nn.Softmax(0)
        #self.log_softmax = nn.LogSoftmax(1)
        #self.softmax_1 = nn.Sot
        #self.softmax = nn.Softmax(0)
    
    def forward(self, logits, region_logits, label_vec):
        prob = self.softmax_1(logits)*self.softmax_0(region_logits)
        prob = prob.sum(0)
        
        loss_pos = -torch.sum(label_vec*torch.log(prob+1e-6)) / (torch.sum(label_vec) + 1e-6)
        loss_neg = -torch.sum((1-label_vec)*torch.log(1-prob+1e-6)) / (torch.sum(1-label_vec) + 1e-6)
        loss = -label_vec*torch.log(prob+1e-6) - (1-label_vec)*torch.log(1-prob+1e-6)
        loss = loss.mean()
        
        return loss_pos, loss_neg, loss


class BatchMILLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mil_criterion = MILLoss()
    
    def forward(self, list_logits, list_region_logits, list_label_vec):
        loss = 0
        loss_pos = 0
        loss_neg = 0
        for logits, region_logits, label_vec in \
            zip(list_logits, list_region_logits, list_label_vec):
            lp,ln,l = self.mil_criterion(logits,region_logits,label_vec)
            loss = loss + l
            loss_pos = loss_pos + lp
            loss_neg = loss_neg + ln

        loss = loss / len(list_logits)
        loss_pos = loss_pos / len(list_logits)
        loss_neg = loss_neg / len(list_logits)
        return loss_pos, loss_neg, loss


class DetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1024,91) #4421
        self.log_softmax = nn.LogSoftmax(1)
    
    def forward(self, features, det_label_vec):
        log_prob = self.log_softmax(self.linear(features))
        loss = -torch.sum(det_label_vec*log_prob,1).mean()
        return loss


class BatchDetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.det_criterion = DetLoss()
        self.softmax = nn.Softmax()
    
    def forward(self, list_features, list_det_label_vec):
        loss = 0
        for features, det_label_vec in zip(list_features, list_det_label_vec):
            loss = loss + self.det_criterion(
                self.softmax(features),det_label_vec)

        loss = loss / len(list_features)
        return loss


def train_model(model,dataloaders,exp_const,tb_writer):
    params = [
        {'params': model.detector.parameters()},
        {'params': model.det_criterion.parameters()},
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

    softmax = nn.Softmax(1)

    best_val_loss = 10000
    for epoch in range(exp_const.num_epochs):
        for it,data in enumerate(dataloaders['train']):
            if data is None:
                continue
            
            # Set mode
            model.detector.train()
            model.det_criterion.train()

            images = [img.cuda() for img in data['image']]
            label_vec = [vec.cuda() for vec in data['label_vec']]
            boxes = [b.cuda() for b in data['boxes']]
            det_label_vec = [s.cuda() for s in data['det_label_vec']]

            logits,region_logits,features = model.detector(images,boxes)
            loss_pos, loss_neg, mil_loss = model.mil_criterion(logits,region_logits,label_vec)
            det_loss = model.det_criterion(features,det_label_vec)
            loss = mil_loss + det_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step%exp_const.log_step==0:
                log_items = {
                    'Loss/Train': loss.item(),
                    'MIL_Loss/Train': mil_loss.item(),
                    'MIL_Pos/Train': loss_pos.item(),
                    'MIL_Neg/Train': loss_neg.item(),
                    'Det_Loss/Train': det_loss.item(),
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
                    'detector': model.detector,
                    'det_criterion': model.det_criterion,
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
    model.det_criterion = BatchDetLoss().cuda()
    model.detector.to_file(os.path.join(exp_const.exp_dir,'detector.txt'))

    train_model(model,dataloaders,exp_const,tb_writer)