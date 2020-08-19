import click
import os
import nltk
from tqdm import tqdm

import utils.io as io
from .dataset_flickr import FlickrDatasetConstants, FlickrDataset, flickr_paths
from .models.cap_encoder import CapEncoderConstants, CapEncoder
from .identify_tokens import (combine_subtokens, align_pos_tokens,
    get_noun_token_ids, group_token_ids, ignore_words_from_pos)


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
    data_const.read_noun_token_ids = False
    dataset = FlickrDataset(data_const)
    noun_token_ids = [None]*len(dataset)
    noun_vocab = set()
    num_human_captions = 0
    num_noun_captions = 0
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
        noun_token_ids_, noun_words = get_noun_token_ids(pos_tags,alignment)
        noun_token_ids_ = group_token_ids(noun_token_ids_,tokens)
        if len(noun_token_ids_) > 0:
            num_noun_captions += 1
        
        noun_token_ids[i] = {
            'image_id': image_id,
            'cap_id': cap_id,
            'token_ids': noun_token_ids_,
            'words': list(noun_words)}
        
        noun_vocab.update(noun_words)
        
        for human_word in ['man','person','human','woman','boy','girl',
            'men','women','boys','girls','child','children']:
            if human_word in tokens:
                num_human_captions += 1
                break

    io.mkdir_if_not_exists(os.path.join(flickr_paths['proc_dir'],'annotations'))
    io.dump_json_object(noun_token_ids,data_const.noun_tokens_json)
    io.dump_json_object(sorted(list(noun_vocab)),data_const.noun_vocab_json)
    print('Number of human captions:',num_human_captions)
    print('Number of noun captions:',num_noun_captions)
    print('Total number of captions:',len(dataset))
    print('Size of noun vocabulary:',len(noun_vocab))


if __name__=='__main__':
    main()