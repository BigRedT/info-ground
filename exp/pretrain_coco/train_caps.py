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
from .cap_encoder import CapEncoder
from .dataset import DetFeatDataset
from .info_nce_loss import InfoNCE
from .factored_cap_info_nce_loss import CapInfoNCE, KLayer, FLayer
#from .cap_info_nce_loss import KVLayer, CapInfoNCE


def create_info_nce_criterion(x_dim,c_dim,d):
    fx = nn.Sequential(
        nn.Linear(x_dim,d))

    fy = nn.Sequential(
        nn.Linear(c_dim,d))

    criterion = InfoNCE(fx,fy)
    
    return criterion


# def create_cap_info_nce_criterion(o_dim,w_dim,d):
#     fo = KVLayer(o_dim,d)
#     fw = KVLayer(w_dim,d)

#     criterion = CapInfoNCE(fo,fw)
    
#     return criterion


def create_cap_info_nce_criterion(o_dim,u_dim,w_dim,d):
    fo = FLayer(o_dim,d)
    fw = FLayer(w_dim,d)
    ku = KLayer(u_dim,d)
    kw = KLayer(w_dim,d)
    criterion = CapInfoNCE(fo,fw,ku,kw)
    
    return criterion


def train_model(model,dataloaders,exp_const,tb_writer):
    # params = itertools.chain(
    #     model.object_encoder.parameters(),
    #     model.self_sup_criterion.parameters())

    params = [
        {'params': model.object_encoder.parameters()},
        {'params': model.self_sup_criterion.parameters()},
        {'params': model.lang_sup_criterion.parameters()},
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
            # Set mode
            model.object_encoder.train()
            model.self_sup_criterion.train()
            model.lang_sup_criterion.train()

            # Forward pass
            object_features = data['features'].cuda()
            object_mask = data['object_mask'].cuda()
            pad_mask = data['pad_mask'].cuda()
            context_object_features = model.object_encoder(
                object_features,
                object_mask,
                pad_mask)
                
            # Computer self supervision loss
            self_sup_loss = model.self_sup_criterion(
                object_features,
                context_object_features,
                object_mask)

            # Compute cap-image loss
            with torch.no_grad():
                token_ids, tokens, token_lens = model.cap_encoder.tokenize_batch(
                    data['caption'])
                token_ids = torch.LongTensor(token_ids).cuda()
                word_features = model.cap_encoder(token_ids)
                noun_token_ids = data['noun_token_ids'].cuda()
                word_features, token_mask = model.cap_encoder.select_noun_embed(
                    word_features,
                    noun_token_ids)

            #token_mask = model.cap_encoder.get_token_mask(tokens)
            #token_mask = torch.cuda.FloatTensor(token_mask)
            lang_sup_loss, att = model.lang_sup_criterion(
                context_object_features,
                object_features,
                word_features.detach(),
                token_mask)

            loss = 0*self_sup_loss + lang_sup_loss

            # Backward pass
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step%exp_const.log_step==0:
                log_items = {
                    'Self_Sup_Loss/Train': self_sup_loss.item(),
                    'Lang_Sup_Loss/Train': lang_sup_loss.item(),
                    'Loss/Train': loss.item(),
                    'Step': step,
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
                    'self_sup_criterion': model.self_sup_criterion,
                    'lang_sup_criterion': model.lang_sup_criterion,
                }

                for name,nn_model in save_items.items():
                    model_path = os.path.join(
                        exp_const.model_dir,
                        f'{name}_{step}')
                    torch.save({
                        'state_dict': nn_model.state_dict(),
                        'step': step,
                        'val_loss': None,
                        'prev_best_val_loss': best_val_loss},
                        model_path)

            if step%exp_const.val_step==0:
                with torch.no_grad():
                    eval_results = eval_model(
                        model,
                        dataloaders['val'],
                        exp_const,
                        step)

                print('Val reults:',eval_results)
                tb_writer.add_scalar(
                    'Self_Sup_Loss/Val',
                    eval_results['self_sup_loss'],
                    step)
                tb_writer.add_scalar(
                    'Lang_Sup_Loss/Val',
                    eval_results['lang_sup_loss'],
                    step)
                tb_writer.add_scalar(
                    'Loss/Val',
                    eval_results['total_loss'],
                    step)
                val_loss = eval_results['total_loss']
                if val_loss < best_val_loss:                    
                    print(f'Saving best model at {step} ...')
                    save_items = {
                        'best_object_encoder': model.object_encoder,
                        'best_self_sup_criterion': model.self_sup_criterion,
                        'best_lang_sup_criterion': model.lang_sup_criterion,
                    }

                    for name,nn_model in save_items.items():
                        model_path = os.path.join(
                            exp_const.model_dir,
                            name)
                        torch.save({
                            'state_dict':nn_model.state_dict(),
                            'step': step,
                            'val_loss': val_loss,
                            'prev_best_val_loss': best_val_loss},
                            model_path)

                    best_val_loss = val_loss

            step += 1


def eval_model(model,dataloader,exp_const,step):
    # Set mode
    model.object_encoder.eval()
    model.self_sup_criterion.eval()
    model.lang_sup_criterion.eval()

    avg_self_sup_loss = 0
    avg_lang_sup_loss = 0
    num_samples = 0
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
            
        # Compute loss
        self_sup_loss = model.self_sup_criterion(
            object_features,
            context_object_features,
            object_mask)  

        token_ids, tokens, token_lens = model.cap_encoder.tokenize_batch(
            data['caption'])
        token_ids = torch.LongTensor(token_ids).cuda()
        word_features = model.cap_encoder(token_ids)
        noun_token_ids = data['noun_token_ids'].cuda()
        word_features, token_mask = model.cap_encoder.select_noun_embed(
            word_features,
            noun_token_ids)

        #token_mask = model.cap_encoder.get_token_mask(tokens)
        #token_mask = torch.cuda.FloatTensor(token_mask)
        lang_sup_loss, att = model.lang_sup_criterion(
            context_object_features,
            object_features,
            word_features,
            token_mask)

        # Aggregate loss or accuracy
        batch_size = object_features.size(0)
        num_samples += batch_size
        avg_self_sup_loss += (self_sup_loss.item()*batch_size)
        avg_lang_sup_loss += (lang_sup_loss.item()*batch_size)

    avg_self_sup_loss = avg_self_sup_loss / num_samples
    avg_lang_sup_loss = avg_lang_sup_loss / num_samples
    total_loss = 0*avg_self_sup_loss + avg_lang_sup_loss

    eval_results = {
        'self_sup_loss': avg_self_sup_loss, 
        'lang_sup_loss': avg_lang_sup_loss,
        'total_loss': total_loss,
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
    model.cap_encoder = CapEncoder(model.const.cap_encoder)
    model.self_sup_criterion = create_info_nce_criterion(
        model.object_encoder.const.object_feature_dim,
        model.object_encoder.const.context_layer.hidden_size,
        model.object_encoder.const.context_layer.hidden_size)
    model.lang_sup_criterion = create_cap_info_nce_criterion(
        model.object_encoder.const.context_layer.hidden_size,
        model.object_encoder.const.object_feature_dim,
        model.cap_encoder.model.config.hidden_size,
        model.cap_encoder.model.config.hidden_size//2)
    if model.const.model_num != -1:
        model.object_encoder.load_state_dict(
            torch.load(model.const.object_encoder_path)['state_dict'])
        model.self_sup_criterion.load_state_dict(
            torch.load(model.const.self_sup_criterion_path)['state_dict'])
        model.lang_sup_criterion.load_state_dict(
            torch.load(model.const.lang_sup_criterion_path)['state_dict'])
    model.object_encoder.cuda()
    model.cap_encoder.cuda()
    model.self_sup_criterion.cuda()
    model.lang_sup_criterion.cuda()
    model.object_encoder.to_file(
        os.path.join(exp_const.exp_dir,'object_encoder.txt'))
    model.self_sup_criterion.to_file(
        os.path.join(exp_const.exp_dir,'self_supervised_criterion.txt'))
    model.lang_sup_criterion.to_file(
        os.path.join(exp_const.exp_dir,'lang_supervised_criterion.txt'))

    print('Creating dataloader ...')
    dataloaders = {}
    for mode, const in data_const.items():
        dataset = DetFeatDataset(const)
        
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
            num_workers=exp_const.num_workers)

    train_model(model,dataloaders,exp_const,tb_writer)