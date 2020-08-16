import os
import h5py
import math
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import itertools
import numpy as np

import utils.io as io
from utils.constants import save_constants, Constants
from .models.object_encoder import ObjectEncoder
from .models.cap_encoder import CapEncoder
from .models.factored_cap_info_nce_loss import CapInfoNCE, KLayer, FLayer
from exp.eval_flickr.dataset import FlickrDataset
from utils.bbox_utils import compute_iou, point_in_box, compute_center
from .train import create_cap_info_nce_criterion


# def create_cap_info_nce_criterion(o_dim,u_dim,w_dim,d):
#     fo = FLayer(o_dim,d)
#     fw = FLayer(w_dim,d)
#     ku = KLayer(u_dim,d)
#     kw = KLayer(w_dim,d)
#     criterion = CapInfoNCE(fo,fw,ku,kw)
    
#     return criterion


# def combine_token_obj_att(token_obj_att,tokens):
#     num_tokens = len(tokens)
#     att = []
#     special_chars = {'-',"'"}
#     for i in range(1,num_tokens-1):
#         token = tokens[i]
#         if (len(token) >= 2 and token[:2]=='##') or \
#             (tokens[i-1] in special_chars) or (token in special_chars):
#             att[-1] = torch.max(att[-1],token_obj_att[i])
#         else:
#             att.append(token_obj_att[i])
    
#     att = torch.stack(att,0)
#     return att


def combine_tokens(tokens,words):
    num_tokens = len(tokens)
    att = []
    combined_tokens = []
    i = 1
    j = 0
    while(i < num_tokens-1):
        token = tokens[i]
        word = words[j].lower()
        if token==word:
            combined_tokens.append({i})
            j+=1
            i+=1
        
        else:
            idx_set = {i}
            prev_token=tokens[i]
            i+=1
            while(prev_token!=word):
                idx_set.add(i)
                token = tokens[i]
                if len(token) > 2 and token[:2]=='##':
                    token = token[2:]

                prev_token = prev_token + token
                i+=1
            
            combined_tokens.append(idx_set)
            j+=1
    
    return combined_tokens
        

def map_phrase_to_tokens(phrase_info,combined_tokens):
    phrase = phrase_info['phrase']
    num_words = len(phrase.split())
    start_idx = phrase_info['first_word_index']
    token_ids = set()
    for i in range(num_words):
        token_ids.update(combined_tokens[start_idx+i])

    return token_ids


def combine_att(token_obj_att,phrase_token_ids):
    phrase_att = None
    for i in phrase_token_ids:
        if phrase_att is None:
            phrase_att = token_obj_att[i]
        else:
            phrase_att = torch.max(phrase_att,token_obj_att[i])
    
    return phrase_att


def select_boxes(boxes,phrase_att,k=3):
    box_ids = torch.topk(phrase_att,k)[1]
    selected_boxes = [boxes[i] for i in box_ids if i < len(boxes)]
    return selected_boxes


def compute_recall(pred_boxes,gt_boxes,k=1):
    recalled = [0]*k
    pred_box = [None]*k
    gt_box = [None]*k
    for i,pred_box_ in enumerate(pred_boxes):
        if i>=k:
            break

        for gt_box_ in gt_boxes:
            iou = compute_iou(pred_box_,gt_box_)
            if iou >= 0.5:
                recalled[i] = 1
                pred_box[i] = pred_box_
                gt_box[i] = gt_box_
                break
    
    max_recall = 0
    for i in range(k):
        max_recall = max(recalled[i],max_recall)
        recalled[i] = max_recall

    return recalled, pred_box, gt_box


def compute_pt_acc(pred_boxes,gt_boxes):
    pt_recalled = False
    for i,pred_box_ in enumerate(pred_boxes):
        if i>=1:
            break
        
        pred_center = compute_center(pred_box_)
        for gt_box_ in gt_boxes:
            pt_recalled = point_in_box(pred_center,gt_box_)
            if pt_recalled is True:
                break
            
    return float(pt_recalled)


def eval_model(model,dataset,exp_const):
    # Set mode
    model.object_encoder.eval()
    model.cap_encoder.eval()
    model.lang_sup_criterion.eval()

    pt_recalled_phrases = 0
    Ks = [1,5,10]
    recalled_phrases = [0]*3
    num_phrases = 0
    for it,data in enumerate(tqdm(dataset)):
        # Forward pass
        object_features = torch.FloatTensor(data['features']).cuda().unsqueeze(0)
        pad_mask = torch.FloatTensor(data['pad_mask']).cuda().unsqueeze(0)

        if exp_const.contextualize==True:
            context_object_features, obj_obj_att = model.object_encoder(
                object_features,
                pad_mask=pad_mask)
        else:
            context_object_features = object_features

        # Compute loss 
        token_ids, tokens, token_lens = model.cap_encoder.tokenize_batch(
            [data['caption']])
        token_ids = torch.LongTensor(token_ids).cuda()
        token_features, word_word_att = model.cap_encoder(token_ids)

        token_mask = torch.zeros(token_ids.size()).cuda()
        lang_sup_loss, token_obj_att, att_V_o = \
            model.lang_sup_criterion(
                context_object_features,
                object_features,
                token_features,
                token_mask)

        token_obj_att = token_obj_att[0,0]
        tokens = tokens[0]
        words = data['caption'].split()
        combined_tokens = combine_tokens(tokens,words)
        
        for phrase_info in data['phrases']:
            phrase_token_ids = map_phrase_to_tokens(
                phrase_info,
                combined_tokens)

            phrase_att = combine_att(token_obj_att,phrase_token_ids)
            pred_boxes = select_boxes(data['boxes'],phrase_att,k=Ks[-1])
            
            phrase_id = phrase_info['phrase_id']
            if phrase_id not in data['gt_boxes']['boxes']:
                continue

            gt_phrase_boxes = data['gt_boxes']['boxes'][phrase_id]

            is_recalled, pred_box, gt_box = compute_recall(
                pred_boxes,
                gt_phrase_boxes,
                k=Ks[-1])
        
            for i in range(3):
                recalled_phrases[i] += is_recalled[Ks[i]-1]
            
            is_pt_recalled = compute_pt_acc(pred_boxes,gt_phrase_boxes)
            pt_recalled_phrases += is_pt_recalled

            num_phrases += 1

            if num_phrases%500 == 0:
                recall = [rp/num_phrases for rp in recalled_phrases]
                pt_recall = pt_recalled_phrases / num_phrases
                print(recall,pt_recall)
            #import pdb; pdb.set_trace()
    
    recall = [round(100*rp/num_phrases,2) for rp in recalled_phrases]
    pt_recall = round(100*pt_recalled_phrases / num_phrases,2)
    results = {
        'recall': {},
        'pt_recall': None
    }
    for i,k in enumerate(Ks):
        results['recall'][k] = recall[i]
    
    results['pt_recall'] = pt_recall
    print(results)

    return results


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
    
    o_dim = model.object_encoder.const.object_feature_dim
    if exp_const.contextualize==True:
        o_dim = model.object_encoder.const.context_layer.hidden_size
    
    model.lang_sup_criterion = create_cap_info_nce_criterion(
        o_dim,
        model.object_encoder.const.object_feature_dim,
        model.cap_encoder.model.config.hidden_size,
        model.cap_encoder.model.config.hidden_size//2,
        model.const.cap_info_nce_layers)
    if model.const.model_num != -1:
        loaded_object_encoder = torch.load(model.const.object_encoder_path)
        print('Loaded model number:',loaded_object_encoder['step'])
        model.object_encoder.load_state_dict(
            loaded_object_encoder['state_dict'])
        model.lang_sup_criterion.load_state_dict(
            torch.load(model.const.lang_sup_criterion_path)['state_dict'])
        if exp_const.random_lang is True:
            model.cap_encoder.load_state_dict(
                torch.load(model.const.cap_encoder_path)['state_dict'])

    model.object_encoder.cuda()
    model.cap_encoder.cuda()
    model.lang_sup_criterion.cuda()

    print('Creating dataloader ...')
    dataset = FlickrDataset(data_const)

    with torch.no_grad():
        results = eval_model(model,dataset,exp_const)

    filename = os.path.join(
        exp_const.exp_dir,
        f'results_{data_const.subset}_{model_const.model_num}.json')
    io.dump_json_object(results,filename)