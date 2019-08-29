import os
import h5py
import click
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
def main(**kwargs):
    print('Creating dataloader ...')
    dataset = DetInputDataset(kwargs['det_input'])
    dataloader = DataLoader(
        dataset,
        batch_size=kwargs['batch'],
        num_workers=kwargs['workers'],
        collate_fn=dataset.create_collate_fn())

    print('Creating detector ...')
    model = create_detector()
    model = model.cuda()
    model.eval()

    print('Run detector on dataset ...')
    for data in dataloader:
        images = [img.cuda() for img in data['img']]
        detections = model(images)
        for detection in detections:
            import pdb; pdb.set_trace()


if __name__=='__main__':
    main()