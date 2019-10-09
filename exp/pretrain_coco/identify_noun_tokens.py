import click
import os
import nltk
from tqdm import tqdm

import utils.io as io
from .dataset import DetFeatDatasetConstants
from .cap_encoder import CapEncoderConstants, CapEncoder


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

def get_noun_token_ids(pos_tags,alignment):
    token_ids = []
    for i, (word,tag) in enumerate(pos_tags):
        if tag in ['NN','NNS','NNP','NNPS']:
            for idx in alignment[i]:
                token_ids.append(idx)
            
    return token_ids


@click.command()
@click.option(
    '--subset',
    type=click.Choice(['train','val','test']),
    default='coco subset to identify nouns for')
def main(**kwargs):
    print('Creating Caption Encoder (tokenizer) ...')
    cap_encoder = CapEncoder(CapEncoderConstants())

    nltk.download('punkt')

    data_const = DetFeatDatasetConstants(kwargs['subset'])
    annos = io.load_json_object(data_const.annos_json)['annotations']
    noun_token_ids = [None]*len(annos)
    for i,anno in enumerate(tqdm(annos)):
        image_id = anno['image_id']
        cap_id = anno['id']
        caption = anno['caption']
        token_ids, tokens = cap_encoder.tokenize(caption)
        nltk_tokens = nltk.word_tokenize(caption.lower())
        pos_tags = nltk.pos_tag(nltk_tokens)
        alignment = align_pos_tokens(pos_tags,tokens)
        noun_token_ids[i] = {
            'image_id': image_id,
            'cap_id': cap_id,
            'token_ids': get_noun_token_ids(pos_tags,alignment)}

    io.dump_json_object(noun_token_ids,data_const.noun_tokens_json)


if __name__=='__main__':
    main()