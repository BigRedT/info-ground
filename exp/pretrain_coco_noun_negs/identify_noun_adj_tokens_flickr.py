import click
import os
import nltk
from tqdm import tqdm

import utils.io as io
from .dataset_flickr import FlickrDatasetConstants, FlickrDataset
from .models.cap_encoder import CapEncoderConstants, CapEncoder


def combine_subtokens(tokens,i):
    ## assumes subtoken starts at i
    count = 1
    word = tokens[i]
    for j in range(i+1,len(tokens)):
        token = tokens[j]
        if len(token)>2 and token[:2]=='##':
            count += 1 
            word += token[2:]
        else:
            break
    
    return word, count


def align_pos_tokens(pos_tags,tokens):
    alignment = [None]*len(pos_tags)
    for i in range(len(alignment)):
        alignment[i] = []
    
    token_len = len(tokens)
    last_match = -1
    skip_until = -1
    for i, (word,tag) in enumerate(pos_tags):
        for j in range(last_match+1,token_len):
            if j < skip_until:
                continue
            
            if j==skip_until:
                skip_until = -1

            token = tokens[j]
            if word==token:
                alignment[i].append(j)
                last_match = j
                break
            elif len(token)>2 and token[:2]=='##':
                combined_token, sub_token_count = combine_subtokens(tokens,j-1)
                skip_until = j-1+sub_token_count
                if word==combined_token:
                    for k in range(sub_token_count):
                        alignment[i].append(k+j-1)
                        last_match = j-1+sub_token_count-1
                 
    return alignment


def get_noun_adj_token_ids(pos_tags,alignment):
    token_ids = []
    for i, (word,tag) in enumerate(pos_tags):
        if tag in ['NN','NNS','NNP','NNPS','JJ','JJR','JJS']:
            for idx in alignment[i]:
                token_ids.append(idx)
            
    return token_ids


def get_noun_token_ids(pos_tags,alignment):
    token_ids = []
    for i, (word,tag) in enumerate(pos_tags):
        if tag in ['NN','NNS','NNP','NNPS']:
            for idx in alignment[i]:
                token_ids.append(idx)
            
    return token_ids


def ignore_words_from_pos(pos_tags,words_to_ignore):
    for i in range(len(pos_tags)):
        word, tag = pos_tags[i]
        if word in words_to_ignore:
            pos_tags[i] = (word,'IG')
        
    return pos_tags


@click.command()
@click.option(
    '--subset',
    type=click.Choice(['train','val','test']),
    default='coco subset to identify nouns for')
def main(**kwargs):
    print('Creating Caption Encoder (tokenizer) ...')
    cap_encoder = CapEncoder(CapEncoderConstants())

    nltk.download('punkt')

    data_const = FlickrDatasetConstants(kwargs['subset'])
    data_const.read_noun_verb_tokens = False
    data_const.read_neg_noun_samples = False
    dataset = FlickrDataset(data_const)
    noun_adj_token_ids = [None]*len(dataset)
    for i,data in enumerate(tqdm(dataset)):
        image_id = data['image_id']
        cap_id = data['cap_id']
        caption = data['caption']
        token_ids, tokens = cap_encoder.tokenize(caption)
        
        nltk_tokens = nltk.word_tokenize(caption.lower())
        pos_tags = nltk.pos_tag(nltk_tokens)
        pos_tags = ignore_words_from_pos(
            pos_tags,['is','has','have','had','be'])
        
        alignment = align_pos_tokens(pos_tags,tokens)
        
        noun_adj_token_ids_ = get_noun_adj_token_ids(pos_tags,alignment)
        noun_adj_tokens_ = []
        for k in noun_adj_token_ids_:
            noun_adj_tokens_.append(tokens[k])

        noun_adj_token_ids[i] = {
            'image_id': image_id,
            'cap_id': cap_id,
            'token_ids': noun_adj_token_ids_,
            'tokens': noun_adj_tokens_}

    io.dump_json_object(noun_adj_token_ids,data_const.noun_verb_tokens_json)


if __name__=='__main__':
    main()