import os
import glob
import click
from tqdm import tqdm

from data.coco.constants import coco_paths,CocoConstants
import utils.io as io


@click.command()
@click.option(
    '--out_dir',
    type=str,
    default=coco_paths['proc_dir'],
    help='Output directory')
@click.option(
    '--subset',
    type=click.Choice(['train','val','test']),
    default='train',
    help='Subset to run detection on')
def main(**kwargs):
    data_const = CocoConstants()
    image_dir = data_const.image_subset_dir[kwargs['subset']]
    image_path_list = glob.glob(os.path.join(image_dir,'*.jpg'))
    det_input = []
    for image_path in tqdm(image_path_list):
        det_input.append({
            'path': image_path,
            'id': os.path.splitext(os.path.basename(image_path))[0]
        })

    io.dump_json_object(
        det_input,
        os.path.join(
            kwargs['out_dir'],
            'det_input_'+kwargs['subset']+'.json'))


if __name__=='__main__':
    main()
