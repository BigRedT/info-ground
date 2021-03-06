import os
import h5py
import click
import torch
import copy
from tqdm import tqdm

from global_constants import flickr_paths
import utils.io as io
from utils.html_writer import HtmlWriter
from .models.cap_encoder import CapEncoderConstants, CapEncoder
from .cache_neg_features import convert_tokens_to_ids, remove_padding


@click.command()
@click.option(
    '--subset',
    type=click.Choice(['train','val','test']),
    default='flickr subset to identify nouns for')
def main(**kwargs):
    model_const = CapEncoderConstants()
    cap_encoder = CapEncoder(model_const).cuda()

    subset = kwargs['subset']

    filename = os.path.join(
        flickr_paths['proc_dir'],
        flickr_paths['noun_negatives']['samples'][subset])
    neg_samples = io.load_json_object(filename)

    filename = os.path.join(
        flickr_paths['proc_dir'],
        flickr_paths['noun_negatives']['feats'][subset])
    feats_f = h5py.File(filename,'w')

    total_count = 0
    diff_len_count = 0
    for image_id in tqdm(neg_samples.keys()):
        for cap_id in neg_samples[image_id].keys():
            for str_neg_idx in neg_samples[image_id][cap_id]['negs'].keys():
                neg_idx = int(str_neg_idx)
                
                pos_tokens = remove_padding(neg_samples[image_id][cap_id]['gt'])
                pos_batch_tokens = [pos_tokens]
                pos_token_ids = convert_tokens_to_ids(pos_batch_tokens,cap_encoder)
                pos_token_ids = torch.LongTensor(pos_token_ids).cuda()
                pos_feats = cap_encoder(pos_token_ids)
                pos_feats = pos_feats[:,neg_idx,:]

                neg_batch_tokens = \
                    neg_samples[image_id][cap_id]['negs'][str_neg_idx]
                neg_token_ids = convert_tokens_to_ids(neg_batch_tokens,cap_encoder)
                neg_token_ids = torch.LongTensor(neg_token_ids).cuda()
                neg_feats = cap_encoder(neg_token_ids)
                neg_feats = neg_feats[:,neg_idx,:]

                feats = torch.cat((pos_feats,neg_feats),0)
                feats = feats.cpu().detach().numpy()
                feats_f.create_dataset(
                    f'{image_id}_{cap_id}_{str_neg_idx}',
                    data=feats)

    feats_f.close()


if __name__=='__main__':
    with torch.no_grad():
        main()