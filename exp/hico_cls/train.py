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
#import sklearn.metrics as metrics

import utils.io as io
from utils.constants import save_constants, Constants
from .models.object_encoder import ObjectEncoder
from .models.hoi_classifier import HOIClassifier
from .models.bce_loss import BCELoss, BalancedBCELoss
from .dataset import HICOFeatDataset
from .compute_mAP import compute_mAP_given_neg_labels


def train_model(model,dataloaders,exp_const,tb_writer):
    params = [
        {'params': model.object_encoder.parameters()},
        {'params': model.hoi_classifier.parameters()},
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

    scheduler = optim.lr_scheduler.StepLR(opt,step_size=20,gamma=0.1)

    if model.const.model_num==-1:
        step = 0
        best_val_mAP = 0
    else:
        step = model.const.model_num
        best_val_mAP = exp_const.best_val_mAP

    
    for epoch in range(exp_const.num_epochs):
        for it,data in enumerate(dataloaders['train']):
            # Set mode
            model.object_encoder.train()
            model.hoi_classifier.train()

            # Forward pass
            object_features = data['features'].cuda()
            object_mask = data['object_mask'].cuda()
            pad_mask = data['pad_mask'].cuda()
            context_object_features = model.object_encoder(
                object_features,
                object_mask,
                pad_mask)

            if exp_const.finetune_object_encoder==False or \
                (exp_const.warmup==True and epoch < 20):
                context_object_features = context_object_features.detach()
            
            context_object_features = torch.cat((
                object_features,context_object_features),2)

            hoi_logits, hoi_context_object_features = \
                model.hoi_classifier(
                    context_object_features,
                    object_mask,
                    pad_mask)
                
            # Compute HOI loss
            pos_labels = data['pos_labels'].cuda()
            neg_labels = data['neg_labels'].cuda()
            unk_labels = data['unk_labels'].cuda()
            
            if exp_const.ignore_unk_labels_during_training==True:
                train_neg_labels = neg_labels
            else:
                train_neg_labels = neg_labels+unk_labels
            
            bce_loss = model.bce_criterion(
                hoi_logits,
                pos_labels,
                train_neg_labels)
            
            loss = bce_loss

            # Backward pass
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step%exp_const.log_step==0:
                log_items = {
                    'BCE_Loss/Train': bce_loss.item(),
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
                    'object_encoder': model.object_encoder,
                    'hoi_classifier': model.hoi_classifier
                    }

                for name,nn_model in save_items.items():
                    model_path = os.path.join(
                        exp_const.model_dir,
                        f'{name}_{step}')
                    torch.save({
                        'state_dict': nn_model.state_dict(),
                        'step': step,
                        'val_mAP': None,
                        'prev_best_val_mAP': best_val_mAP},
                        model_path)

            if step%exp_const.val_step==0:
                with torch.no_grad():
                    eval_results = eval_model(
                        model,
                        dataloaders['val'],
                        exp_const,
                        step)

                #print('Val reults:',eval_results)
                log_items = {
                    'BCE_Loss/Val': eval_results['bce_loss'],
                    'Loss/Val': eval_results['total_loss'],
                    'mAP/Val': eval_results['mAP'],
                    'Max_AP/Val': max(eval_results['APs']),
                    'Min_AP/Val': min(eval_results['APs']),
                }
                print('Val results:',log_items)
                
                for name,value in log_items.items():
                    tb_writer.add_scalar(name,value,step)

                val_mAP = eval_results['mAP']
                if val_mAP > best_val_mAP:                    
                    print(f'Saving best model at {step} ...')
                    save_items = {
                        'best_object_encoder': model.object_encoder,
                        'best_hoi_classifier': model.hoi_classifier,
                    }

                    for name,nn_model in save_items.items():
                        model_path = os.path.join(
                            exp_const.model_dir,
                            name)
                        torch.save({
                            'state_dict':nn_model.state_dict(),
                            'step': step,
                            'val_mAP': val_mAP,
                            'prev_best_val_mAP': best_val_mAP},
                            model_path)

                    best_val_mAP = val_mAP

            step += 1

        scheduler.step()


def eval_model(model,dataloader,exp_const,step):
    # Set mode
    model.object_encoder.eval()
    model.hoi_classifier.eval()

    avg_bce_loss = 0
    num_samples = 0
    list_of_pos_labels = []
    list_of_neg_labels = []
    list_of_unk_labels = []
    list_of_pred = []
    for it,data in enumerate(tqdm(dataloader)):
        if (exp_const.num_val_samples is not None) and \
            (num_samples >= exp_const.num_val_samples):
                break

        # Forward pass
        object_features = data['features'].cuda()
        object_mask = data['object_mask'].cuda()
        pad_mask = data['pad_mask'].cuda()
        context_object_features = model.object_encoder(
            object_features,
            object_mask,
            pad_mask)
            
        context_object_features = torch.cat((
            object_features,context_object_features),2)

        hoi_logits, hoi_context_object_features = \
            model.hoi_classifier(
                context_object_features,
                object_mask,
                pad_mask)
                
        # Compute HOI loss
        pos_labels = data['pos_labels'].cuda()
        neg_labels = data['neg_labels'].cuda()
        unk_labels = data['unk_labels'].cuda()
        bce_loss = model.bce_criterion(
            hoi_logits,
            pos_labels,
            neg_labels+unk_labels)

        list_of_pos_labels.append(pos_labels.detach().cpu().numpy())
        list_of_neg_labels.append(neg_labels.detach().cpu().numpy())
        list_of_unk_labels.append(unk_labels.detach().cpu().numpy())
        list_of_pred.append(hoi_logits.detach().cpu().numpy())

        # Aggregate loss or accuracy
        batch_size = object_features.size(0)
        num_samples += batch_size
        avg_bce_loss += (bce_loss.item()*batch_size)

    avg_bce_loss = avg_bce_loss / num_samples
    total_loss = avg_bce_loss

    all_pos_labels = np.concatenate(list_of_pos_labels,0)
    all_neg_labels = np.concatenate(list_of_neg_labels,0)
    all_unk_labels = np.concatenate(list_of_unk_labels,0)
    all_pred = np.concatenate(list_of_pred,0)
    mAP, APs = compute_mAP_given_neg_labels(
        y_true=all_pos_labels,
        y_false=all_neg_labels+all_unk_labels,
        y_score=all_pred)

    eval_results = {
        'bce_loss': avg_bce_loss,
        'total_loss': total_loss,
        'mAP': mAP,
        'APs': APs,
    }

    return eval_results


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
        f'data_val_{model_num}': data_const['val'],
        f'model_{model_num}': model_const},
        exp_const.exp_dir)
    
    print('Creating network ...')
    model = Constants()
    model.const = model_const
    model.object_encoder = ObjectEncoder(model.const.object_encoder)
    model.hoi_classifier = HOIClassifier(model.const.hoi_classifier)
    if exp_const.balanced_bce==True:
        model.bce_criterion = BalancedBCELoss()
    else:
        model.bce_criterion = BCELoss()
    if exp_const.pretrained_object_encoder_path!='unavailable':
        print('Loading a pretrained object encoder ...')
        model.object_encoder.load_state_dict(
            torch.load(exp_const.pretrained_object_encoder_path)['state_dict'])
    
    if model.const.model_num != -1:
        print('Loading a specified model number ...')
        loaded_model = torch.load(exp_const.pretrained_object_encoder_path)
        model.object_encoder.load_state_dict(
            loaded_model['state_dict'])
        model.hoi_classifier.load_state_dict(
            torch.load(model.const.hoi_classifier_path)['state_dict'])
        
        val_mAP = loaded_model['val_mAP']
        prev_best_val_mAP = loaded_model['prev_best_val_mAP']
        if val_mAP is None:
            exp_const.best_val_mAP = prev_best_val_mAP
        else:
            exp_const.best_val_mAP = max(val_mAP,prev_best_val_mAP)
        
    model.object_encoder.cuda()
    model.hoi_classifier.cuda()
    model.object_encoder.to_file(
        os.path.join(exp_const.exp_dir,'object_encoder.txt'))
    model.hoi_classifier.to_file(
        os.path.join(exp_const.exp_dir,'hoi_classifier.txt'))

    print('Creating dataloader ...')
    dataloaders = {}
    for mode, const in data_const.items():
        dataset = HICOFeatDataset(const)
        
        if mode=='train':
            shuffle=True
            batch_size=exp_const.train_batch_size
        else:
            shuffle=False
            batch_size=exp_const.val_batch_size

        dataloaders[mode] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=exp_const.num_workers)

    train_model(model,dataloaders,exp_const,tb_writer)