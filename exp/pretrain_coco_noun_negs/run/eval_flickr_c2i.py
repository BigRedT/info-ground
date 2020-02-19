import os
import click

from utils.constants import Constants, ExpConstants
from global_constants import coco_paths
from exp.eval_flickr.dataset import  FlickrDatasetConstants
from exp.eval_flickr.self_sup_dataset import SelfSupFlickrDatasetConstants
from ..models.object_encoder import ObjectEncoderConstants
from ..models.cap_encoder import CapEncoderConstants
from .. import eval_flickr_c2i

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
@click.option(
    '--no_context',
    is_flag=True,
    help='Apply flag to switch off contextualization')
@click.option(
    '--self_sup_feat',
    is_flag=True,
    help='Apply flag to use self-supervised features')
def main(**kwargs):
    exp_const = ExpConstants(kwargs['exp_name'],kwargs['exp_base_dir'])
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.seed = 0
    exp_const.contextualize = not kwargs['no_context']
    exp_const.self_sup_feat = kwargs['self_sup_feat']

    DatasetConstants = FlickrDatasetConstants
    if exp_const.self_sup_feat==True:
        DatasetConstants = SelfSupFlickrDatasetConstants

    data_const = DatasetConstants('test')

    model_const = Constants()
    model_const.model_num = kwargs['model_num']
    model_const.object_encoder = ObjectEncoderConstants()
    model_const.object_encoder.context_layer.output_attentions = True
    model_const.object_encoder.object_feature_dim = 2048
    if exp_const.self_sup_feat==True:
        model_const.object_encoder.object_feature_dim = 2048 + 256
    model_const.cap_encoder = CapEncoderConstants()
    model_const.cap_encoder.output_attentions = True

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

    eval_flickr_c2i.main(exp_const,data_const,model_const)


if __name__=='__main__':
    main()