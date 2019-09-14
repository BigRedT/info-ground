import os
import click
from tqdm import tqdm

import utils.io as io
from detector.model import COCO_INSTANCE_CATEGORY_NAMES
from global_constants import coco_paths


@click.command()
@click.option(
    '--labels_hdf5',
    type=str,
    help='Path to labels')
@click.option(
    '--object_index_json',
    type=str,
    help='Path to object index json relative to proc_dir')
def main(**kwargs):
    os.environ['HDF5_USE_FILE_LOCKING']="FALSE"
    f = io.load_h5py_object(kwargs['labels_hdf5'])
    
    num_objects = len(COCO_INSTANCE_CATEGORY_NAMES)
    object_index = [None]*num_objects
    for i in range(num_objects):
        object_index[i] = []

    for image_id in tqdm(f.keys()):
        labels = f[image_id][()]
        for i,label in enumerate(labels):
            object_index[label].append((image_id,i))

    filepath = os.path.join(coco_paths['proc_dir'],kwargs['object_index_json'])
    io.dump_json_object(object_index,filepath)


if __name__=='__main__':
    main()