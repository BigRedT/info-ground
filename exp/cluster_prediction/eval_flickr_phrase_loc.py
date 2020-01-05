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

from detector.model import create_detector
from .models.cap_encoder import CapEncoder
from .cluster_labeler import ClusterLabeler
from exp.eval_flickr.dataset_wo_features import FlickrDataset
from utils.bbox_utils import compute_iou, point_in_box, compute_center


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


def map_phrase_last_word_to_tokens(phrase_info,combined_tokens):
    phrase = phrase_info['phrase']
    num_words = len(phrase.split())
    return combined_tokens[phrase_info['first_word_index']+num_words-1]


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


def eval_model(model,dataset,cluster_labeler,exp_const):
    model.detector.eval()
    model.cap_encoder.eval()

    pt_recalled_phrases = 0
    recalled_phrases = [0]*3
    num_phrases = 0
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        
        boxes = [torch.FloatTensor(data['boxes'][:20]).cuda()]
        image = [torch.FloatTensor(data['image']).cuda()]
        _,region_logits,_ = model.detector(image,boxes)
        region_logits = region_logits[0]
        _,selected_box_ids = torch.max(region_logits,0)
        selected_box_ids = selected_box_ids.detach().cpu().numpy()

        token_ids, tokens, token_lens= model.cap_encoder.tokenize_batch([data['caption']])
        token_embed = model.cap_encoder(torch.LongTensor(token_ids).cuda())[0]
        tokens = tokens[0]
        words = data['caption'].split(' ')
        combined_tokens = combine_tokens(tokens,words)
        for phrase_info in data['phrases']:
            phrase_token_ids = map_phrase_last_word_to_tokens(
                phrase_info,
                combined_tokens)
            phrase_embed = token_embed[list(phrase_token_ids)]
            phrase_embed = phrase_embed.mean(0).detach().cpu().numpy()
            word = phrase_info['phrase'].split(' ')[-1]
            label = cluster_labeler.get_label(word,phrase_embed)
            idx = cluster_labeler.get_idx(label)
            if idx==-1:
                pred_boxes = [data['boxes'][0]]
            else:
                box_id = selected_box_ids[idx]
                pred_boxes = [data['boxes'][box_id]]

            phrase_id = phrase_info['phrase_id']
            if phrase_id not in data['gt_boxes']['boxes']:
                continue

            gt_phrase_boxes = data['gt_boxes']['boxes'][phrase_id]

            is_recalled, pred_box, gt_box = compute_recall(
                pred_boxes,
                gt_phrase_boxes,
                k=1)

            for k in range(1):
                recalled_phrases[k] += is_recalled[k]
            
            is_pt_recalled = compute_pt_acc(pred_boxes,gt_phrase_boxes)
            pt_recalled_phrases += is_pt_recalled

            num_phrases += 1

            if num_phrases%500 == 0:
                recall = [rp/num_phrases for rp in recalled_phrases]
                pt_recall = pt_recalled_phrases / num_phrases
                print(recall,pt_recall)
    
    recall = [round(100*rp/num_phrases,2) for rp in recalled_phrases]
    pt_recall = round(100*pt_recalled_phrases / num_phrases,2)
    print(recall,pt_recall)
    
    import pdb; pdb.set_trace()



def main(exp_const,data_const,model_const):
    np.random.seed(exp_const.seed)
    torch.manual_seed(exp_const.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print('Creating dataloader ...')
    dataset = FlickrDataset(data_const)
    
    print('Loading cluster info ...')
    print('    Loading cluster centers ...')
    cluster_centers = np.load(os.path.join(
        data_const.cluster_info['dir'],
        data_const.cluster_info['centers']))

    print('    Loading labels ...')
    labels = io.load_json_object(os.path.join(
        data_const.cluster_info['dir'],
        data_const.cluster_info['labels']['train']))

    print('    Loading active nouns ...')
    active_nouns = io.load_json_object(os.path.join(
        data_const.cluster_info['dir'],
        data_const.cluster_info['active_nouns']))

    print('Creating cluster labeler ...')
    cluster_labeler = ClusterLabeler(
        cluster_centers,
        labels,
        active_nouns)

    print('Creating network ...')
    model = Constants()
    model.const = model_const
    model.const.num_classes = len(labels)
    print('Num classes:',model.const.num_classes)
    model.cap_encoder = CapEncoder(model.const.cap_encoder)
    model.detector = create_detector(
        extractor=True,
        num_classes=model.const.num_classes).cuda()
    loaded_detector = torch.load(model.const.detector_path)
    model.detector.load_state_dict(loaded_detector['state_dict'])
    model.cap_encoder.cuda()
    model.detector.cuda()

    with torch.no_grad():
        recall = eval_model(model,dataset,cluster_labeler,exp_const)