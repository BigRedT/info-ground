import os
import h5py
import click
import torch
import copy
from tqdm import tqdm

from global_constants import coco_paths
import utils.io as io
from utils.html_writer import HtmlWriter
from .models.cap_encoder import CapEncoderConstants, CapEncoder


def convert_tokens_to_ids(batch_tokens,cap_encoder):
    converter = cap_encoder.tokenizer._convert_token_to_id

    batch_token_ids = []
    for tokens in batch_tokens:
        token_ids = []
        for token in tokens:
            token_ids.append(converter(token))
        
        batch_token_ids.append(token_ids)
    
    return batch_token_ids
    

@click.command()
@click.option(
    '--subset',
    type=click.Choice(['train','val','test']),
    default='coco subset to identify nouns for')
def main(**kwargs):
    model_const = CapEncoderConstants()
    cap_encoder = CapEncoder(model_const).cuda()

    subset = kwargs['subset']

    filename = os.path.join(
        coco_paths['proc_dir'],
        coco_paths['extracted']['negatives']['samples'][subset])
    neg_samples = io.load_json_object(filename)

    filename = os.path.join(
        coco_paths['proc_dir'],
        coco_paths['extracted']['negatives']['feats'][subset])
    feats_f = h5py.File(filename,'w')

    for image_id in tqdm(neg_samples.keys()):
        for cap_id in neg_samples[image_id].keys():
            for str_neg_idx in neg_samples[image_id][cap_id]['negs'].keys():
                neg_idx = int(str_neg_idx)
                batch_tokens = \
                    neg_samples[image_id][cap_id]['negs'][str_neg_idx]
                token_ids = convert_tokens_to_ids(batch_tokens,cap_encoder)
                token_ids = torch.LongTensor(token_ids).cuda()
                feats = cap_encoder(token_ids)
                feats = feats[:,neg_idx,:]
                feats = feats.cpu().detach().numpy()
                feats_f.create_dataset(
                    f'{image_id}_{cap_id}_{str_neg_idx}',
                    data=feats)

    feats_f.close()


if __name__=='__main__':
    main()