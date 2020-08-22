import os
import click

from utils.constants import Constants, ExpConstants
from global_constants import coco_paths


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
    exp_const.optimizer = 'Adam'
    exp_const.lr = 1e-3
    exp_const.momentum = None
    exp_const.num_epochs = 100
    exp_const.log_step = 100
    exp_const.model_save_step = 1000
    exp_const.val_step = 1000
    exp_const.num_val_samples = None

    data_const = {
        'train': Constants(),
        'val': Constants()
    }

    model_const = Constants()
    model_const.model_num = kwargs['model_num']
    model_const.net = Constants()
    model_const.net_path = os.path.join(
        exp_const.model_dir,
        f'net_{model_const.model_num}')
