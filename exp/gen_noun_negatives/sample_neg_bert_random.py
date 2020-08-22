import os
import click
import torch
import copy
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

from global_constants import coco_paths
import utils.io as io
from utils.html_writer import HtmlWriter
from .dataset import DetFeatDatasetConstants, DetFeatDataset
from .models.cap_encoder import CapEncoderConstants, CapEncoder


def replace_tokens(token_ids,ids_to_repl,repl_val):
    token_ids = copy.deepcopy(token_ids)
    B = len(ids_to_repl)
    for i in range(B):
        for j in ids_to_repl[i]:
            token_ids[i][j] = repl_val
        
    return token_ids


def insert_word(tokens, noun_token_ids, word):
    new_tokens = None
    noun_idx = -1
    if len(noun_token_ids)==0:
        new_tokens = tokens
    else:
        i = noun_token_ids[0]
        j = noun_token_ids[-1]
        new_tokens = tokens[:i] + [word] + tokens[j+1:]
        noun_idx = i

    return new_tokens, noun_idx


def tokens_to_sentence(tokens):
    clean_tokens = []
    for token in tokens:
        if token in ['[CLS]','[SEP]','[PAD]']:
            continue

        if len(token)>=2 and token[:2]=='##':
            clean_tokens[-1] += token[2:]
        elif token in ['.',',']:
            clean_tokens[-1] += token
        else:
            clean_tokens.append(token)
    
    return ' '.join(clean_tokens)


def remove_pad(tokens):
    clean_tokens = []
    for i,token in enumerate(tokens):
        if token=='[PAD]':
            continue
        
        clean_tokens.append(token)
    
    return clean_tokens

        
def get_id_to_token_converter(cap_encoder):
    def id_to_token_converter(idx):
        return cap_encoder.tokenizer._convert_id_to_token(idx)
    
    return id_to_token_converter


def create_vocab_mask(id_to_token_converter,model_vocab_size,noun_vocab):
    mask = torch.zeros([1,1,model_vocab_size],dtype=torch.float32)
    noun_vocab = set(noun_vocab)
    for i in range(model_vocab_size):
        token = id_to_token_converter(i)
        if token in noun_vocab:
            mask[0,0,i] = 1

    return mask


def apply_vocab_mask(logits,vocab_mask):
    logits = -10000*(1-vocab_mask) + logits
    prob = logits.softmax(-1)
    return prob


def sort_by_scores(score_preds):
    return sorted(score_preds,key=lambda x: x[1],reverse=True)


def ensemble_prediction(token_ids,cap_encoder,T=5):
    agg_pred = None
    for t in range(T):
        masked_token_ids = cap_encoder.mask_batch(token_ids)
        pred = cap_encoder.model(masked_token_ids)[0]
        if agg_pred is None:
            agg_pred = pred
        else:
            agg_pred = agg_pred + pred

    agg_pred = agg_pred / T
    return agg_pred


class RandomCaptionSampler():
    def __init__(self,dataset,k):
        self.annos = dataset.annos['annotations']
        self.annos = list(zip(self.annos,range(len(self.annos))))
        self.noun_token_ids = dataset.noun_token_ids
        self.k = k

    def sample(self,image_id):
        candidates = random.sample(self.annos,2*self.k)
        samples = []
        for cand,i in candidates:
            if len(samples)==self.k:
                break

            if str(cand['image_id'])!=str(image_id):
                selected_noun_tokens = []
                if len(self.noun_token_ids[i]['token_ids']) > 0:
                    selected_noun_tokens = random.choice(
                        self.noun_token_ids[i]['token_ids'])
                
                noun_idx = -1
                if len(selected_noun_tokens) > 0:
                    noun_idx = selected_noun_tokens[0]

                cand['noun_token_idx'] = noun_idx
                samples.append(cand)
            
        return samples
    
    def batch(self,samples):
        noun_token_ids = []
        captions = []
        for s in samples:
            noun_token_ids.append(s['noun_token_idx'])
            captions.append(s['caption'])
        
        return noun_token_ids, captions


@click.command()
@click.option(
    '--subset',
    type=click.Choice(['train','val','test']),
    default='coco subset to identify nouns for')
@click.option(
    '--select',
    type=int,
    default=5,
    help='Number of samples to select')
def main(**kwargs):
    model_const = CapEncoderConstants()
    model_const.model = 'BertForPreTraining'
    cap_encoder = CapEncoder(model_const).cuda()
    id_to_token_converter = get_id_to_token_converter(cap_encoder)

    subset = kwargs['subset']
    const = DetFeatDatasetConstants(subset)
    const.read_noun_tokens = True
    dataset = DetFeatDataset(const)
    print(len(dataset))
    collate_fn = dataset.get_collate_fn()
    dataloader = DataLoader(dataset,20,num_workers=10,collate_fn=collate_fn)

    random_cap_sampler = RandomCaptionSampler(dataset,kwargs['select'])

    noun_vocab = io.load_json_object(const.noun_vocab_json)
    vocab_mask = create_vocab_mask(
        id_to_token_converter,
        cap_encoder.tokenizer.vocab_size,
        noun_vocab)

    neg_samples = {}

    for it,data in enumerate(tqdm(dataloader)):
        noun_token_ids = data['noun_token_ids']
        token_ids_, tokens, token_lens = cap_encoder.tokenize_batch(
            data['caption'])
        mask_token_ids_ = replace_tokens(
            token_ids_,noun_token_ids,cap_encoder.mask_token_id)
        token_ids = torch.LongTensor(token_ids_)
        mask_token_ids = torch.LongTensor(mask_token_ids_)
        
        B = token_ids.size(0)
        for i in range(B):
            if len(noun_token_ids[i])==0:
                continue

            j = noun_token_ids[i][0]

            image_id = str(data['image_id'][i].item())
            cap_id = str(data['cap_id'][i].item())

            if image_id not in neg_samples:
                neg_samples[image_id] = {}

            if cap_id not in neg_samples[image_id]:
                neg_samples[image_id][cap_id] = {
                    'gt': tokens[i],
                    'negs': {}}
            
            neg_noun_token_ids, neg_captions = random_cap_sampler.batch(
                random_cap_sampler.sample(data['image_id'][i].item()))

            _, neg_tokens, _ = cap_encoder.tokenize_batch(
                neg_captions)
            _,neg_idx = insert_word(
                tokens[i],
                noun_token_ids[i],
                '_NONE_')
            str_neg_idx = str(neg_idx)

            neg_samples[image_id][cap_id]['negs'][str_neg_idx] = {
                'neg_tokens': neg_tokens,
                'neg_token_ids': neg_noun_token_ids}


    filename = os.path.join(
        coco_paths['proc_dir'],
        coco_paths['extracted']['noun_negatives']['samples'][subset])
    io.dump_json_object(neg_samples,filename)



if __name__=='__main__':
    with torch.no_grad():
        main()      