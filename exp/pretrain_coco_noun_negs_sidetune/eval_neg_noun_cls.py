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
from .models.object_encoder import ObjectEncoder
from .models.cap_encoder import CapEncoder
from .models.info_nce_loss import InfoNCE
from .models.factored_cap_info_nce_loss import CapInfoNCE, KLayer, FLayer
from .models.neg_verb_loss import compute_neg_verb_loss
from .dataset import DetFeatDataset


def create_info_nce_criterion(x_dim,c_dim,d):
    fx = nn.Sequential(
        nn.Linear(x_dim,d))

    fy = nn.Sequential(
        nn.Linear(c_dim,d))

    criterion = InfoNCE(fx,fy)
    
    return criterion


def create_cap_info_nce_criterion(o_dim,u_dim,w_dim,d):
    fo = FLayer(o_dim,d)
    fw = FLayer(w_dim,d)
    ku = KLayer(u_dim,d)
    kw = KLayer(w_dim,d)
    criterion = CapInfoNCE(fo,fw,ku,kw)
    
    return criterion


def eval_model(model,dataloader,exp_const):
    # Set mode
    model.object_encoder.eval()
    model.lang_sup_criterion.eval()

    total_correct = 0
    total_samples = 0
    for it,data in enumerate(tqdm(dataloader)):
        if (exp_const.num_val_samples is not None) and \
            (total_samples >= exp_const.num_val_samples):
                break

        # Forward pass
        object_features = data['features'].cuda()
        object_mask = data['object_mask'].cuda()
        pad_mask = data['pad_mask'].cuda()
        context_object_features, obj_obj_att = model.object_encoder(
            object_features,
            object_mask,
            pad_mask)
            
        # Compute loss 
        token_ids, tokens, token_lens = model.cap_encoder.tokenize_batch(
            data['caption'])
        token_ids = torch.LongTensor(token_ids).cuda()
        token_features, word_word_att = model.cap_encoder(token_ids)
        noun_verb_token_ids = data['noun_verb_token_ids'].cuda()
        word_features, token_mask = model.cap_encoder.select_embed(
            token_features,
            noun_verb_token_ids)
        noun_ids = data['noun_id'].cuda()
        _, noun_token_mask = model.cap_encoder.select_embed(
            token_features,
            noun_ids.unsqueeze(1))

        noun_feats = data['neg_noun_feats'].cuda()

        att_V_o = model.lang_sup_criterion.att_V_o_for_verbs(
            context_object_features,
            object_features,
            noun_feats.detach()) # Bx(N+1)xD
        
        valid_noun_mask = 1-noun_token_mask # Bx1
        neg_noun_loss,log_softmax = compute_neg_verb_loss(
            att_V_o,
            noun_feats,
            valid_noun_mask,
            model.lang_sup_criterion.fw.f_layer)

        pred_noun = torch.argmax(log_softmax,1).float()
        pred_noun = pred_noun*valid_noun_mask[:,0] + -1*(1-valid_noun_mask[:,0])
        correct = (pred_noun==0).sum().item()
        samples = torch.sum(valid_noun_mask[:,0]).item()
        total_correct += correct
        total_samples += samples

        if it%500==0:
            noun_acc = round(100*total_correct / total_samples,2)
            print('Iter:',it,'noun Acc:',noun_acc)

    noun_acc = total_correct / total_samples

    eval_results = {
        'noun_acc': noun_acc,
    }

    return eval_results


def main(exp_const,data_const,model_const):
    np.random.seed(exp_const.seed)
    torch.manual_seed(exp_const.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print('Creating network ...')
    model = Constants()
    model.const = model_const
    model.object_encoder = ObjectEncoder(model.const.object_encoder)
    model.cap_encoder = CapEncoder(model.const.cap_encoder)
    model.lang_sup_criterion = create_cap_info_nce_criterion(
        model.object_encoder.const.context_layer.hidden_size,
        model.object_encoder.const.object_feature_dim,
        model.cap_encoder.model.config.hidden_size,
        model.cap_encoder.model.config.hidden_size//2)
    if model.const.model_num != -1:
        model.object_encoder.load_state_dict(
            torch.load(model.const.object_encoder_path)['state_dict'])
        model.lang_sup_criterion.load_state_dict(
            torch.load(model.const.lang_sup_criterion_path)['state_dict'])
    model.object_encoder.cuda()
    model.cap_encoder.cuda()
    model.lang_sup_criterion.cuda()

    print('Creating dataloader ...')
    dataset = DetFeatDataset(data_const)
    dataloader = DataLoader(
        dataset,
        batch_size=exp_const.batch_size,
        shuffle=False,
        num_workers=exp_const.num_workers)

    with torch.no_grad():
        eval_results = eval_model(model,dataloader,exp_const)

    print(eval_results)