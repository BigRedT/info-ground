import os
import click

from utils.constants import Constants, ExpConstants
from global_constants import coco_paths
from ..dataset import DetFeatDatasetConstants
from ..models.object_encoder import ObjectEncoderConstants
from ..models.cap_encoder import CapEncoderConstants
from ..vis_word_word_att import main as vis_word_word_att


@click.command()
@click.option(
    '--exp_name',
    default='default_exp',
    help='Experiment name')
@click.option(
    '--exp_base_dir',
    default=coco_paths['exp_dir'],
    help='Output directory where a folder would be created for each experiment')
def main(**kwargs):
    exp_const = ExpConstants(kwargs['exp_name'],kwargs['exp_base_dir'])
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'logs')
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.sinkhorn = False

    vis_dir_name = 'vis/word_word_attention_soft'
    if exp_const.sinkhorn==True:
        vis_dir_name += '_sinkhorn'
        
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,vis_dir_name)
    exp_const.num_vis_samples = 20
    exp_const.seed = 0

    data_const = DetFeatDatasetConstants('val')
    data_const.image_dir = os.path.join(
        coco_paths['image_dir'],
        data_const.subset_image_dirname)
    data_const.read_noun_tokens = True

    model_const = Constants()
    model_const.cap_encoder = CapEncoderConstants()
    model_const.cap_encoder.output_attentions = True

    vis_word_word_att(exp_const,data_const,model_const)


if __name__=='__main__':
    main()