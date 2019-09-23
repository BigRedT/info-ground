import os
import click

from utils.constants import Constants, ExpConstants
from global_constants import coco_paths
from ..dataset import DetFeatDatasetConstants
from ..object_encoder import ObjectEncoderConstants
from ..cache_context_features import main as cache_context_features


@click.command()
@click.option(
    '--exp_name',
    default='default_exp',
    help='Experiment name')
@click.option(
    '--exp_base_dir',
    default=coco_paths['exp_dir'],
    help='Output directory where a folder would be created for each experiment')
@click.option(
    '--encoder_path',
    type=str,
    help='Context encoder path')
@click.option(
    '--features_hdf5',
    type=str,
    help='Path to the object feature hdf5 file to be contextualized')
@click.option(
    '--context_features_hdf5',
    type=str,
    help='Name of context features hdf5 file relative to exp_dir')
def main(**kwargs):
    exp_const = ExpConstants(kwargs['exp_name'],kwargs['exp_base_dir'])
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
    
    data_const = Constants()
    data_const.features_hdf5 = kwargs['features_hdf5']
    data_const.context_features_hdf5 = kwargs['context_features_hdf5']

    model_const = Constants()
    model_const.object_encoder = ObjectEncoderConstants()
    model_const.object_encoder_path = kwargs['encoder_path']

    cache_context_features(exp_const,data_const,model_const)


if __name__=='__main__':
    main()