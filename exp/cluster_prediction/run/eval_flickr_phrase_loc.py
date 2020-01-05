import os
import click

from utils.constants import Constants, ExpConstants
from global_constants import coco_paths
from exp.eval_flickr.dataset_wo_features import  FlickrDatasetConstants
from ..models.cap_encoder import CapEncoderConstants
from .. import eval_flickr_phrase_loc

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
    '--model_num',
    default=-1,
    type=int,
    help='Model number. -1 implies begining of training. -100 means best')
def main(**kwargs):
    exp_const = ExpConstants(kwargs['exp_name'],kwargs['exp_base_dir'])
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.seed = 0

    data_const = FlickrDatasetConstants('test')
    data_const.cluster_info = coco_paths['clusters']

    model_const = Constants()
    model_const.model_num = kwargs['model_num']
    model_const.cap_encoder = CapEncoderConstants()
    model_const.detector_path = os.path.join(
        exp_const.model_dir,
        f'detector_{model_const.model_num}')

    eval_flickr_phrase_loc.main(exp_const,data_const,model_const)


if __name__=='__main__':
    main()