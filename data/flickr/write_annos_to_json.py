import os
import click
from tqdm import tqdm

import utils.io as io
from .constants import FlickrConstants
from .flickr30k_entities_utils import get_annotations, get_sentence_data


@click.command()
@click.option(
    '--subset',
    type=str,
    default='test',
    help='Subset to preprocess')
def main(**kwargs):
    subset = kwargs['subset']
    const = FlickrConstants()
    
    io.mkdir_if_not_exists(const.flickr_paths['proc_dir'])
    
    image_ids = io.read(const.subset_ids[subset])
    image_ids = [idx.decode() for idx in image_ids.split()]
    
    # Write boxes to json
    boxes = {}
    for image_id in tqdm(image_ids):
        box_xml = os.path.join(const.flickr_paths['anno_dir'],f'{image_id}.xml')
        boxes[image_id] = get_annotations(box_xml)

    io.dump_json_object(boxes,const.box_json[subset])
    
    # Write sentence annos to json
    sent = {}
    for image_id in tqdm(image_ids):
        sent_txt = os.path.join(
            const.flickr_paths['sent_dir'],
            f'{image_id}.txt')
        sent[image_id] = get_sentence_data(sent_txt)

    io.dump_json_object(sent,const.sent_json[subset])


if __name__=='__main__':
    main()