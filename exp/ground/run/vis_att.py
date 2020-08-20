import os
import click

from utils.constants import Constants, ExpConstants
from global_constants import coco_paths, flickr_paths
from ..dataset import DetFeatDatasetConstants as CocoDatasetConstants
from exp.eval_flickr.dataset import FlickrDatasetConstants
from ..models.object_encoder import ObjectEncoderConstants
from ..models.cap_encoder import CapEncoderConstants
from ..vis_att import main as vis_att
from ..vis_att_flickr import main as vis_att_flickr


@click.command()
@click.option(
    '--exp_name',
    default='default_exp',
    help='Experiment name')
@click.option(
    '--train_dataset',
    default='coco',
    type=click.Choice(['coco','flickr']),
    help='Dataset to use')
@click.option(
    '--vis_dataset',
    default='coco',
    type=click.Choice(['coco','flickr']),
    help='Dataset to use')
@click.option(
    '--no_context',
    is_flag=True,
    help='Apply flag to switch off contextualization')
@click.option(
    '--model_num',
    default=-1,
    type=int,
    help='Model number. -1 implies begining of training. -100 means best')
def main(**kwargs):
    exp_base_dir = coco_paths['exp_dir']
    if kwargs['train_dataset']=='flickr':
        exp_base_dir = flickr_paths['exp_dir']
    exp_const = ExpConstants(kwargs['exp_name'],exp_base_dir)
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'logs')
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.train_dataset = kwargs['train_dataset']
    exp_const.vis_dataset = kwargs['vis_dataset']
    exp_const.vis_dir = os.path.join(
        exp_const.exp_dir,
        f'vis/attention_{exp_const.vis_dataset}')
    exp_const.num_vis_samples = 50
    exp_const.seed = 0
    exp_const.contextualize = not kwargs['no_context']

    DatasetConstants = CocoDatasetConstants
    if exp_const.vis_dataset=='flickr':
        DatasetConstants = FlickrDatasetConstants

    data_const = DatasetConstants('val') 
    if exp_const.vis_dataset=='coco':
        data_const.image_dir = os.path.join(
            coco_paths['image_dir'],
            data_const.subset_image_dirname)
    data_const.read_neg_samples = False
    data_const.read_noun_adj_tokens = False

    model_const = Constants()
    model_const.model_num = kwargs['model_num']
    model_const.object_encoder = ObjectEncoderConstants()
    model_const.object_encoder.object_feature_dim = 2048
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

    if exp_const.vis_dataset=='coco':
        vis_att(exp_const,data_const,model_const)
    else:
        vis_att_flickr(exp_const,data_const,model_const)

if __name__=='__main__':
    main()