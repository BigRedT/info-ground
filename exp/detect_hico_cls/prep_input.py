import os
import click
from tqdm import tqdm

from global_constants import hico_paths
import utils.io as io


@click.command()
@click.option(
    '--out_dir',
    type=str,
    default=hico_paths['proc_dir'],
    help='Output directory')
@click.option(
    '--subset',
    type=click.Choice(['train','val','test']),
    default='train',
    help='Subset to run detection on')
def main(**kwargs):
    subset = kwargs['subset']
    subset_list_json = os.path.join(
        hico_paths['proc_dir'],
        hico_paths['subset_list_json'][subset])
    subset_list = io.load_json_object(subset_list_json)
    image_dir = os.path.join(
        hico_paths['image_dir'],
        hico_paths['image_sub_dir'][subset])
    
    det_input = []
    for image_name in tqdm(subset_list):
        image_path = os.path.join(image_dir,image_name)
        det_input.append({
            'path': image_path,
            'id': image_name,
        })

    io.dump_json_object(
        det_input,
        os.path.join(
            kwargs['out_dir'],
            'det_input_'+kwargs['subset']+'.json'))


if __name__=='__main__':
    main()
