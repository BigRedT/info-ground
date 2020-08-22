import click
import os
import nltk
from tqdm import tqdm

import utils.io as io
from .dataset_flickr import FlickrDatasetConstants, FlickrDataset
from .models.cap_encoder import CapEncoderConstants, CapEncoder
from exp.gen_noun_negatives.identify_tokens import (combine_subtokens,
    align_pos_tokens, ignore_words_from_pos)
from .identify_noun_adj_tokens import get_noun_adj_token_ids


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
    data_const.read_noun_adj_tokens = False
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

    io.dump_json_object(noun_adj_token_ids,data_const.noun_adj_tokens_json)


if __name__=='__main__':
    main()