import os
import click
from tqdm import tqdm

import utils.io as io
from data.flickr.constants import FlickrConstants


@click.command()
@click.option(
    '--out_dir',
    type=str,
    default=FlickrConstants.flickr_paths['proc_dir'],
    help='Output directory')
@click.option(
    '--subset',
    type=click.Choice(['train','val','test']),
    default='test',
    help='Subset to run detection on')
def main(**kwargs):
    subset = kwargs['subset']
    const = FlickrConstants()
    
    image_ids = io.read(const.subset_ids[subset])
    image_ids = [idx.decode() for idx in image_ids.split()]

    image_dir = const.flickr_paths['image_dir']

    det_input = []
    for image_id in tqdm(image_ids):
        image_path = os.path.join(image_dir,f'{image_id}.jpg')
        det_input.append({
            'path': image_path,
            'id': image_id
        })
    
    io.dump_json_object(
        det_input,
        os.path.join(
            kwargs['out_dir'],
            'det_input_'+kwargs['subset']+'.json'))


if __name__=='__main__':
    main()