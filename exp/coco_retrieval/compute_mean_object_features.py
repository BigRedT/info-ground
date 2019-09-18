import os
import click
import h5py
import numpy as np
from tqdm import tqdm

import utils.io as io

@click.command()
@click.option(
    '--feat_h5py',
    type=str,
    help='Path to object features h5py')
@click.option(
    '--out_feat_h5py',
    type=str,
    help='Path to output features h5py')
@click.option(
    '--concat_with_object',
    is_flag=True)
def main(**kwargs):
    os.environ['HDF5_USE_FILE_LOCKING']="FALSE"

    feat_f = io.load_h5py_object(kwargs['feat_h5py'])
    out_feat_f = h5py.File(kwargs['out_feat_h5py'],'w')

    for img_id in tqdm(feat_f.keys()):
        feat = feat_f[img_id][()]
        num_objects = feat.shape[0]
        if num_objects > 0:
            mean_feat = np.mean(feat,0,keepdims=True)
            mean_feat = np.tile(mean_feat,[num_objects,1])
            if kwargs['concat_with_object'] is True:
                mean_feat = np.concatenate([feat,mean_feat],1)
        else:
            mean_feat = feat

        out_feat_f.create_dataset(img_id,data=mean_feat)

    feat_f.close()
    out_feat_f.close()


if __name__=='__main__':
    main()