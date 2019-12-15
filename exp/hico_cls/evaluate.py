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
from .models.bce_loss import BCELoss
from .dataset import HICOFeatDataset
from .compute_mAP import compute_mAP, compute_mAP_given_neg_labels


def eval_model(model,dataloader,exp_const):
    # Set mode
    model.object_encoder.eval()
    model.hoi_classifier.eval()

    sigmoid_layer = nn.Sigmoid()

    avg_bce_loss = 0
    num_samples = 0
    list_of_pos_labels = []
    list_of_neg_labels = []
    list_of_unk_labels = []
    list_of_pred = []
    for it,data in enumerate(tqdm(dataloader)):
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
        hoi_probs = sigmoid_layer(hoi_logits)
                
        # Compute HOI loss
        pos_labels = data['pos_labels'].cuda()
        neg_labels = data['neg_labels'].cuda()
        unk_labels = data['unk_labels'].cuda()
        bce_loss = model.bce_criterion(hoi_logits,pos_labels,neg_labels)

        list_of_pos_labels.append(pos_labels.detach().cpu().numpy())
        list_of_neg_labels.append(neg_labels.detach().cpu().numpy())
        list_of_unk_labels.append(unk_labels.detach().cpu().numpy())
        list_of_pred.append(hoi_probs.detach().cpu().numpy())

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
    # mAP, APs = compute_mAP(
    #     y_true=all_pos_labels,
    #     y_score=all_pred)

    subset = dataloader.dataset.const.subset
    np.save(os.path.join(exp_const.exp_dir,f'prob_{subset}.npy'),all_pred)
    np.save(
        os.path.join(exp_const.exp_dir,f'pos_labels_{subset}.npy'),
        all_pos_labels)

    mAP, APs = compute_mAP_given_neg_labels(
        y_true=all_pos_labels,
        y_false=all_neg_labels + all_unk_labels,
        y_score=all_pred)

    mAP_KO, APs_KO = compute_mAP_given_neg_labels(
        y_true=all_pos_labels,
        y_false=all_neg_labels,
        y_score=all_pred)

    eval_results = {
        'bce_loss': avg_bce_loss,
        'total_loss': total_loss,
        'mAP': mAP,
        'mAP_KO': mAP_KO,
        'APs': APs,
        'APs_KO': APs_KO,
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
    model.hoi_classifier = HOIClassifier(model.const.hoi_classifier)
    model.bce_criterion = BCELoss()
    
    if model.const.model_num != -1:
        print('Loading a specified model number ...')
        model.object_encoder.load_state_dict(
            torch.load(model.const.object_encoder_path)['state_dict'])
        model.hoi_classifier.load_state_dict(
            torch.load(model.const.hoi_classifier_path)['state_dict'])
        
    model.object_encoder.cuda()
    model.hoi_classifier.cuda()

    print('Creating dataloader ...')
    dataset = HICOFeatDataset(data_const)
    dataloader = DataLoader(
        dataset,
        batch_size=exp_const.batch_size,
        shuffle=False,
        num_workers=exp_const.num_workers)

    with torch.no_grad():
        eval_results = eval_model(model,dataloader,exp_const)

    mAP = round(eval_results['mAP']*100,2)
    max_mAP = round(max(eval_results['APs'])*100,2)
    min_mAP = round(min(eval_results['APs'])*100,2)
    print('mAP:',mAP,'max_mAP:',max_mAP,'min_mAP:',min_mAP)

    mAP_KO = round(eval_results['mAP_KO']*100,2)
    max_mAP_KO = round(max(eval_results['APs_KO'])*100,2)
    min_mAP_KO = round(min(eval_results['APs_KO'])*100,2)
    print('mAP_KO:',mAP_KO,'max_mAP_KO:',max_mAP_KO,'min_mAP_KO:',min_mAP_KO)
    
    if model.const.model_num==-100:
        filename = 'eval_results_best.json' 
    else:
        filename = 'eval_results_' + str(model_num) + '.json'

    io.dump_json_object(
        eval_results,
        os.path.join(exp_const.exp_dir,filename))