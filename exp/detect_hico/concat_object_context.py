import h5py
import click
import numpy as np
from tqdm import tqdm

import utils.io as io

@click.command()
@click.option(
    '--obj_feat_hdf5',
    type=str,
    help='Path to object features')
@click.option(
    '--context_feat_hdf5',
    type=str,
    help='Path to context features')
@click.option(
    '--concat_feat_hdf5',
    type=str,
    help='Path to concatenated features')
def main(**kwargs):
    obj_f = io.load_h5py_object(kwargs['obj_feat_hdf5'])
    context_f = io.load_h5py_object(kwargs['context_feat_hdf5'])
    concat_f = h5py.File(kwargs['concat_feat_hdf5'],'w')
    for global_id in tqdm(obj_f.keys()):
        obj_feat = obj_f[global_id][()]
        context_feat = context_f[global_id][()]
        concat_feat = np.concatenate((obj_feat,context_feat),1)
        concat_f.create_dataset(global_id,data=concat_feat)

    obj_f.close()
    context_f.close()
    concat_f.close()


if __name__=='__main__':
    main()