import os
import click
import h5py
import torch
import numpy as np
from tqdm import tqdm

import utils.io as io
from global_constants import coco_paths
from .models.cap_encoder import CapEncoderConstants, CapEncoder


def get_token_feature(features,word_token_ids):
    if len(word_token_ids)==0:
        feat = None
    elif len(word_token_ids)==1:
        feat = features[word_token_ids[0]]
    else:
        feat = np.mean(features[word_token_ids],0)
    
    return feat


@click.command()
@click.option(
    '--subset',
    default='train',
    help='Subset of data to extract features from')
def main(**kwargs):
    subset = kwargs['subset']
    
    print('Reading captions ...')
    captions_json = os.path.join(
        coco_paths['proc_dir'],
        coco_paths['extracted']['annos']['captions'][subset])
    
    print('Reading noun tokens ...')
    noun_tokens_json = os.path.join(
        coco_paths['proc_dir'],
        coco_paths['extracted']['annos']['noun_tokens'][subset])

    print('Aggregate caption and token data into a single list ...')
    captions_data = io.load_json_object(captions_json)
    noun_tokens_data = io.load_json_object(noun_tokens_json)
    caption_token_data = [None]*len(noun_tokens_data)
    num_nouns = 0
    for i in tqdm(range(len(noun_tokens_data))):
        captions_item = captions_data['annotations'][i]
        noun_tokens_item = noun_tokens_data[i]
        err_msg = 'Image ids do not match'
        assert(captions_item['image_id']==noun_tokens_item['image_id']), err_msg
        err_msg = 'Caption ids do not match'
        assert(captions_item['id']==noun_tokens_item['cap_id']), err_msg
        caption_token_item = noun_tokens_item
        caption_token_item['caption'] = captions_item['caption']
        caption_token_data[i] = caption_token_item
        num_nouns += len(caption_token_item['words'])

    print('Num nouns:', num_nouns)

    print('Create caption encoder ...')
    cap_encoder = CapEncoder(CapEncoderConstants()).cuda()
    
    print('Create hdf5 file to store features ...')
    feat_h5py = os.path.join(
        coco_paths['local_proc_dir'],
        coco_paths['extracted']['noun_feats']['feats'][subset])
    feat_f = h5py.File(feat_h5py,'w')
    feat_f.create_dataset('features',(num_nouns,768))
    
    feat_info = [None]*num_nouns
    noun_count = 0
    for cap_token_item in tqdm(caption_token_data):
        token_ids, token_words = cap_encoder.tokenize(cap_token_item['caption'])
        features = cap_encoder(torch.LongTensor([token_ids]).cuda())[0]
        features = features.cpu().detach().numpy()
        for word, word_token_ids in \
            zip(cap_token_item['words'],cap_token_item['token_ids']):
            word_feat = get_token_feature(features,word_token_ids)
            no_feat = True
            if word_feat is not None:
                no_feat = False
                feat_f['features'][noun_count,:] = word_feat
            else:
                feat_f['features'][noun_count,:] = 0
            
            feat_info[noun_count] = {
                'image_id': cap_token_item['image_id'],
                'cap_id': cap_token_item['cap_id'],
                'word': word,
                'token_ids': word_token_ids,
                'caption': cap_token_item['caption'],
                'no_feat': no_feat,
            }
            
            noun_count += 1

    feat_info_json = os.path.join(
        coco_paths['proc_dir'],
        coco_paths['extracted']['noun_feats']['feat_info'][subset])
    io.dump_json_object(feat_info,feat_info_json)


if __name__=='__main__':
    with torch.no_grad():
        main()

    

    