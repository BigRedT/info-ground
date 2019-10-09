import os
import click

from utils.constants import Constants, ExpConstants
from global_constants import coco_paths
from ..dataset import DetFeatDatasetConstants
from ..object_encoder import ObjectEncoderConstants
from ..cap_encoder import CapEncoderConstants
from ..vis_att import main as vis_att


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
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'logs')
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis/attention')
    exp_const.num_vis_samples = 50
    exp_const.seed = 0

    data_const = DetFeatDatasetConstants('val')
    data_const.image_dir = os.path.join(
        coco_paths['image_dir'],
        data_const.subset_image_dirname)

    model_const = Constants()
    model_const.model_num = kwargs['model_num']
    model_const.object_encoder = ObjectEncoderConstants()
    model_const.cap_encoder = CapEncoderConstants()

    if model_const.model_num==-100:
        model_const.object_encoder_path = os.path.join(
            exp_const.model_dir,
            f'best_object_encoder')
        model_const.lang_sup_criterion_path = os.path.join(
            exp_const.model_dir,
            f'best_lang_sup_criterion')
    else:
        model_const.object_encoder_path = os.path.join(
            exp_const.model_dir,
            f'object_encoder_{model_const.model_num}')
        model_const.lang_sup_criterion_path = os.path.join(
            exp_const.model_dir,
            f'lang_sup_criterion_{model_const.model_num}')

    vis_att(exp_const,data_const,model_const)


if __name__=='__main__':
    main()