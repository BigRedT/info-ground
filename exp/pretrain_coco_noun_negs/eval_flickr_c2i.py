import os
import h5py
import math
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import itertools
import numpy as np
import nltk

import utils.io as io
from utils.constants import save_constants, Constants
from .models.object_encoder import ObjectEncoder
from .models.cap_encoder import CapEncoder
from .models.factored_cap_info_nce_loss import CapInfoNCE, KLayer, FLayer
from exp.eval_flickr.dataset import FlickrDataset
from exp.eval_flickr.self_sup_dataset import SelfSupFlickrDataset
from .identify_noun_verb_tokens import combine_subtokens, align_pos_tokens, \
    get_noun_verb_token_ids, get_noun_token_ids, ignore_words_from_pos


def create_cap_info_nce_criterion(o_dim,u_dim,w_dim,d):
    fo = FLayer(o_dim,d)
    fw = FLayer(w_dim,d)
    ku = KLayer(u_dim,d)
    kw = KLayer(w_dim,d)
    criterion = CapInfoNCE(fo,fw,ku,kw)
    
    return criterion


def cache_features(model,dataset,exp_const):
    # Set mode
    model.object_encoder.eval()
    model.cap_encoder.eval()
    model.lang_sup_criterion.eval()
    cap_features = {}
    image_features = {}
    #import pdb; pdb.set_trace()
    for i in tqdm(range(len(dataset))):
    #for i in tqdm(range(100)):
        data = dataset[i]
        image_id = data['image_id']
        cap_num = data['cap_num']

        # Object features
        object_features = torch.FloatTensor(data['features']).cuda().unsqueeze(0)
        pad_mask = torch.FloatTensor(data['pad_mask']).cuda().unsqueeze(0)

        if exp_const.contextualize==True:
            context_object_features, obj_obj_att = model.object_encoder(
                object_features,
                pad_mask=pad_mask)
        else:
            context_object_features = object_features

        # Word features
        token_ids, tokens, token_lens = model.cap_encoder.tokenize_batch(
            [data['caption']])
        token_ids = torch.LongTensor(token_ids).cuda()
        token_features, word_word_att = model.cap_encoder(token_ids)

        nltk_tokens = nltk.word_tokenize(data['caption'].lower())
        pos_tags = nltk.pos_tag(nltk_tokens)
        pos_tags = ignore_words_from_pos(
            pos_tags,['is','has','have','had','be'])
        
        alignment = align_pos_tokens(pos_tags,tokens[0])
        noun_verb_token_ids = get_noun_token_ids(pos_tags,alignment)
        token_features = token_features[:,noun_verb_token_ids,:]
        
        cap_id = f'{image_id}_{cap_num}'
        cap_features[cap_id] = token_features
        image_features[image_id] = object_features
        
    return cap_features, image_features


def compute_recall(cap_id,scores,K=[1,5,10]):
    top_scores = sorted(scores,key=lambda x:x[1],reverse=True)
    query_image_id = cap_id.split('_')[0]
    recall = {}
    for k in K:
        recall[k] = 0
        
    for k in K:
        if query_image_id in [image_id for image_id,score in top_scores[:k]]:
            recall[k] = 1
    
    return recall
            

def compute_scores(cap_features,image_features,model):
    scores = {}
    recall = {1:0,5:0,10:0}
    count = 0
    concat_object_features = []
    concat_image_ids = []
    for image_id, object_features in image_features.items():
        concat_object_features.append(object_features)
        concat_image_ids.append(image_id)

    concat_object_features = torch.cat(concat_object_features,0)
    for i, (cap_id,token_features) in enumerate(tqdm(cap_features.items())):
        scores[cap_id] = []
        token_mask = torch.zeros(token_features.size()[:2]).cuda()
        lang_sup_loss, token_obj_att, att_V_o, logits = \
            model.lang_sup_criterion(
                concat_object_features,
                concat_object_features,
                token_features,
                token_mask,
                return_logits=True)
        scores[cap_id] = logits[0].mean(1).detach().cpu().numpy().tolist()
        for k,r in compute_recall(cap_id,zip(concat_image_ids,scores[cap_id])).items():
            recall[k] += r

        count += 1
        if i % 100==0:
            print(
                f'{count}/{len(cap_features)}',
                recall[1]*100/count,
                recall[5]*100/count,
                recall[10]*100/count)
        
    #     for image_id, object_features in image_features.items():
    #         token_mask = torch.zeros(token_features.size()[:2]).cuda()
    #         lang_sup_loss, token_obj_att, att_V_o, logits = \
    #             model.lang_sup_criterion(
    #                 object_features,
    #                 object_features,
    #                 token_features,
    #                 token_mask,
    #                 return_logits=True)
    #         #import pdb; pdb.set_trace()
    #         score = round(logits.mean().item(),4)
    #         #score = round(token_obj_att.mean().item(),4)
    #         scores[cap_id].append((image_id,score))
    
    #     recall += compute_recall(cap_id,scores[cap_id],k=1)
    #     count += 1
    #     if i % 100==0:
    #         print(f'{count}/{len(cap_features)}',recall*100/count)
    
    for k,r in recall.items():
        recall[k] = r*100/count
    print('Recall',recall)
    import pdb; pdb.set_trace()



def main(exp_const,data_const,model_const):
    nltk.download('punkt')

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
        model.cap_encoder.model.config.hidden_size//2)
    if model.const.model_num != -1:
        loaded_object_encoder = torch.load(model.const.object_encoder_path)
        print('Loaded model number:',loaded_object_encoder['step'])
        model.object_encoder.load_state_dict(
            loaded_object_encoder['state_dict'])
        model.lang_sup_criterion.load_state_dict(
            torch.load(model.const.lang_sup_criterion_path)['state_dict'])
    model.object_encoder.cuda()
    model.cap_encoder.cuda()
    model.lang_sup_criterion.cuda()

    print('Creating dataloader ...')
    FeatDataset = FlickrDataset
    if exp_const.self_sup_feat==True:
        FeatDataset = SelfSupFlickrDataset
    dataset = FeatDataset(data_const)
    print(len(dataset))

    with torch.no_grad():
        word_features, image_features = cache_features(model,dataset,exp_const)
        compute_scores(word_features,image_features,model)