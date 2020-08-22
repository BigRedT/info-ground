import os
import click
import torch
import copy
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


@click.command()
@click.option(
    '--subset',
    type=click.Choice(['train','val','test']),
    default='coco subset to identify nouns for')
@click.option(
    '--rank',
    type=int,
    default=10,
    help='Number of samples to rank')
@click.option(
    '--select',
    type=int,
    default=5,
    help='Number of samples to select')
@click.option(
    '--wo_rerank',
    is_flag=True,
    help='set flag to not rerank')
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

    noun_vocab = io.load_json_object(const.noun_vocab_json)
    vocab_mask = create_vocab_mask(
        id_to_token_converter,
        cap_encoder.tokenizer.vocab_size,
        noun_vocab)
    
    if kwargs['wo_rerank'] is True:
        filename = os.path.join(
            coco_paths['proc_dir'],
            f'vis_bert_noun_negatives_wo_rerank_{subset}.html')
    else:
        filename = os.path.join(
            coco_paths['proc_dir'],
            f'vis_bert_noun_negatives_{subset}.html')

    html_writer = HtmlWriter(filename)

    K = kwargs['rank']

    neg_samples = {}

    for it,data in enumerate(tqdm(dataloader)):
        noun_token_ids = data['noun_token_ids']
        token_ids_, tokens, token_lens = cap_encoder.tokenize_batch(
            data['caption'])
        mask_token_ids_ = replace_tokens(
            token_ids_,noun_token_ids,cap_encoder.mask_token_id)
        token_ids = torch.LongTensor(token_ids_)
        mask_token_ids = torch.LongTensor(mask_token_ids_)
        
        predictions = cap_encoder.model(token_ids.cuda())[0].cpu()
        predictions = apply_vocab_mask(predictions,vocab_mask)
        
        mask_predictions = cap_encoder.model(mask_token_ids.cuda())[0].cpu()
        mask_predictions = apply_vocab_mask(mask_predictions,vocab_mask)

        max_ids = torch.topk(predictions,K,2)[1]
        mask_max_ids = torch.topk(mask_predictions,K,2)[1]

        true_prob_mask_max_ids = torch.gather(predictions,2,mask_max_ids)
        lang_prob_mask_max_ids = torch.gather(mask_predictions,2,mask_max_ids)
        if kwargs['wo_rerank'] is True:
            score = lang_prob_mask_max_ids
        else:
            score = lang_prob_mask_max_ids / (true_prob_mask_max_ids+1e-6)

        max_ids = max_ids.detach().numpy()
        mask_max_ids = mask_max_ids.detach().numpy()
        
        B,L,K = max_ids.shape
        for i in range(B):
            if len(noun_token_ids[i])==0:
                continue

            j = noun_token_ids[i][0]
            preds = []
            mask_preds = []
            score_preds = []
            for k in range(K):
                preds.append(
                    (cap_encoder.tokenizer._convert_id_to_token(max_ids[i][j][k]),
                    round(predictions[i][j][max_ids[i][j][k]].item(),3)))
                mask_preds.append(
                    (cap_encoder.tokenizer._convert_id_to_token(mask_max_ids[i][j][k]),
                    round(mask_predictions[i][j][mask_max_ids[i][j][k]].item(),3)))
                score_preds.append(
                    (cap_encoder.tokenizer._convert_id_to_token(mask_max_ids[i][j][k]),
                    round(score[i][j][k].item(),3)))
            
            rerank_preds = sort_by_scores(score_preds)

            if it <= 50:
                html_writer.add_element({0: '-'*10, 1: '-'*100})
                html_writer.add_element({0: 'Original', 1: tokens[i]})
                html_writer.add_element(
                    {0: 'Token Replaced', 1: tokens[i][j]})
                html_writer.add_element({0: 'True Pred', 1: preds})
                html_writer.add_element({0: 'Lang Pred', 1: mask_preds})
                html_writer.add_element({0: 'Rerank Pred', 1: rerank_preds})


            image_id = str(data['image_id'][i].item())
            cap_id = str(data['cap_id'][i].item())

            if image_id not in neg_samples:
                neg_samples[image_id] = {}

            if cap_id not in neg_samples[image_id]:
                neg_samples[image_id][cap_id] = {
                    'gt': tokens[i],
                    'negs': {}}
            
            for k in range(kwargs['select']):
                new_tokens,neg_idx = insert_word(
                    tokens[i],
                    noun_token_ids[i],
                    rerank_preds[k][0])
                new_tokens = remove_pad(new_tokens)
                str_neg_idx = str(neg_idx)
                if str_neg_idx not in neg_samples[image_id][cap_id]['negs']:
                    neg_samples[image_id][cap_id]['negs'][str_neg_idx] = []

                neg_samples[image_id][cap_id]['negs'][str_neg_idx].append(
                    new_tokens)

    filename = os.path.join(
        coco_paths['proc_dir'],
        coco_paths['extracted']['noun_negatives']['samples'][subset])
    io.dump_json_object(neg_samples,filename)

    html_writer.close()



if __name__=='__main__':
    with torch.no_grad():
        main()      