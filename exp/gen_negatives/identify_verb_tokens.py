import click
import os
import nltk
from tqdm import tqdm

import utils.io as io
from .dataset import DetFeatDatasetConstants
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


def get_verb_token_ids(pos_tags,alignment):
    verb_words = set()
    token_ids = []
    for i, (word,tag) in enumerate(pos_tags):
        if tag in ['VB','VBD','VBG','VBN','VBP','VBZ']:
            verb_words.add(word)
            for idx in alignment[i]:
                token_ids.append(idx)

    return token_ids, verb_words


def group_token_ids(token_ids,tokens):
    grouped_ids = []
    group_num = -1
    for token_id in token_ids:
        token = tokens[token_id]
        if len(token)>=2 and token[:2]=='##':
            grouped_ids[-1].append(token_id)
        else:
            grouped_ids.append([token_id])
    
    return grouped_ids


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

    data_const = DetFeatDatasetConstants(kwargs['subset'])
    annos = io.load_json_object(data_const.annos_json)['annotations']
    verb_token_ids = [None]*len(annos)
    verb_vocab = set()
    num_human_captions = 0
    num_verb_captions = 0
    for i,anno in enumerate(tqdm(annos)):
        image_id = anno['image_id']
        cap_id = anno['id']
        caption = anno['caption']
        token_ids, tokens = cap_encoder.tokenize(caption)
        
        nltk_tokens = nltk.word_tokenize(caption.lower())
        pos_tags = nltk.pos_tag(nltk_tokens)
        pos_tags = ignore_words_from_pos(
            pos_tags,['is','has','have','had','be'])
        
        alignment = align_pos_tokens(pos_tags,tokens)
        verb_token_ids_, verb_words = get_verb_token_ids(pos_tags,alignment)
        verb_token_ids_ = group_token_ids(verb_token_ids_,tokens)
        if len(verb_token_ids_) > 0:
            num_verb_captions += 1
        
        # if 'has' in tokens:
        #     print(pos_tags)
        #     print(tokens)
        #     print(verb_token_ids_)
        #     print(verb_words)
        #     import pdb; pdb.set_trace()
        
        verb_token_ids[i] = {
            'image_id': image_id,
            'cap_id': cap_id,
            'token_ids': verb_token_ids_,
            'words': list(verb_words)}
        
        verb_vocab.update(verb_words)
        
        for human_word in ['man','person','human','woman','boy','girl',
            'men','women','boys','girls','child','children']:
            if human_word in tokens:
                num_human_captions += 1
                break

    io.dump_json_object(verb_token_ids,data_const.verb_tokens_json)
    io.dump_json_object(sorted(list(verb_vocab)),data_const.verb_vocab_json)
    print('Number of human captions:',num_human_captions)
    print('Number of verb captions:',num_verb_captions)
    print('Total number of captions:',len(annos))
    print('Size of verb vocabulary:',len(verb_vocab))


if __name__=='__main__':
    main()