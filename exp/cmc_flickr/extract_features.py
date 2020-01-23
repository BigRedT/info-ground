import os
import h5py
import click
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

from .dataset import FlickrDatasetConstants, FlickrDataset
from .models.resnet import MyResNetsCMC
import utils.io as io
from global_constants import misc_paths, flickr_paths

@click.command()
@click.option(
    '--subset',
    default='train',
    help='subset to extract features from')
@click.option(
    '--batch_size',
    default=10,
    help='number of images to process in one batch')
@click.option(
    '--num_workers',
    default=10,
    help='number of workers in dataloader')
def main(**kwargs):
    subset = kwargs['subset']

    print('Creating dataset ...')
    const = FlickrDatasetConstants(subset)
    dataset = FlickrDataset(const)
    dataloader = DataLoader(
        dataset,
        batch_size=kwargs['batch_size'],
        num_workers=kwargs['num_workers'],
        collate_fn=dataset.get_collate_fn())
    
    print('Creating model ...')
    model = MyResNetsCMC(name='resnet50v3')
    model.load_state_dict(torch.load(misc_paths['cmc_model_path'])['model'])
    model = model.cuda()
    
    print('Creating hdf5 file ...')
    filename = os.path.join(
        flickr_paths['local_proc_dir'],
        flickr_paths['self_sup_feats'][subset])
    feat_h5py = h5py.File(filename,'w')
    
    print('Extracting features ...')
    for data in tqdm(dataloader):
        if data is None:
            continue

        crops = data['crops'].float().cuda()
        feat_l, feat_ab = model(crops)
        feat = torch.cat((feat_l,feat_ab),1)
        feat = feat.detach().cpu().numpy()
        start_idx = 0
        for img_id, num_crops in zip(data['image_id'],data['num_crops']):
            feat_ = feat[start_idx:start_idx+num_crops]
            start_idx = start_idx+num_crops
            feat_h5py.create_dataset(img_id,data=feat_)

    feat_h5py.close()


if __name__=='__main__':
    with torch.no_grad():
        main()
