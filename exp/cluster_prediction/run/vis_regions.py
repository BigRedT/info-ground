import os
import click

from utils.constants import Constants, ExpConstants
from global_constants import coco_paths
from ..dataset import ClusterPredDatasetConstants
from ..vis_regions import main as vis


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
    help='Model number. -1 implies begining of training.')
def main(**kwargs):
    exp_const = ExpConstants(kwargs['exp_name'],kwargs['exp_base_dir'])
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'logs')
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
    exp_const.num_vis_samples = 20
    exp_const.seed = 0
    
    data_const = ClusterPredDatasetConstants('train')

    model_const = Constants()
    model_const.model_num = kwargs['model_num']
    model_const.detector_path = os.path.join(
        exp_const.model_dir,
        f'detector_{model_const.model_num}')

    vis(exp_const,data_const,model_const)


if __name__=='__main__':
    main()