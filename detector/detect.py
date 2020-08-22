import os
import h5py
import click
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

import utils.io as io
from .det_input_dataset import DetInputDataset
from .model import create_detector


@click.command()
@click.option(
    '--det_input',
    type=str,
    help='Detector input json file')
@click.option(
    '--out_dir',
    type=str,
    help='Directory where detections are saved')
@click.option(
    '--batch',
    type=int,
    default=5,
    help='Batch size')
@click.option(
    '--workers',
    type=int,
    default=5,
    help='Number of workers')
@click.option(
    '--dataset',
    type=click.Choice(['coco','default','hico']),
    default='coco',
    help='Choose detector configuration according to dataset')
def main(**kwargs):
    print('Creating dataloader ...')
    dataset = DetInputDataset(kwargs['det_input'])
    dataloader = DataLoader(
        dataset,
        batch_size=kwargs['batch'],
        num_workers=kwargs['workers'],
        collate_fn=dataset.create_collate_fn())

    print('Creating detector ...')
    model = create_detector(kwargs['dataset'])
    model = model.cuda()
    model.eval()

    print('Creating hdf5 files for storing detections ...')
    io.mkdir_if_not_exists(kwargs['out_dir'],recursive=True)
    h5py_f = {}
    for name in ['features','scores','boxes','labels']:
        print('-',os.path.join(kwargs['out_dir'],f'{name}.hdf5'))
        h5py_f[name] = h5py.File(
            os.path.join(kwargs['out_dir'],f'{name}.hdf5'),
            'w')

    print('Run detector on dataset ...')
    for data in tqdm(dataloader):
        images = [img.cuda() for img in data['img']]
        with torch.no_grad():
            detections = model(images)
        for img_id,detection in zip(data['img_id'],detections):
            for name, value in detection.items():
                h5py_f[name].create_dataset(
                    img_id,
                    data=value.cpu().detach().numpy())
            
    for f in h5py_f.values():
        f.close()

if __name__=='__main__':
    main()