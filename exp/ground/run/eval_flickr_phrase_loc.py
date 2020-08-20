import os
import click

import utils.io as io
from utils.constants import Constants, ExpConstants
from global_constants import coco_paths, flickr_paths
from exp.eval_flickr.dataset import  FlickrDatasetConstants
from ..models.object_encoder import ObjectEncoderConstants
from ..models.cap_encoder import CapEncoderConstants
from .. import eval_flickr_phrase_loc

@click.command()
@click.option(
    '--exp_name',
    default='default_exp',
    help='Experiment name')
@click.option(
    '--dataset',
    default='coco',
    type=click.Choice(['coco','flickr']),
    help='Dataset to use')
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
    '--subset',
    default='test',
    help='subset to run evaluation on')
@click.option(
    '--random_lang',
    is_flag=True,
    help='Apply flag to randomly initialize and train BERT')
@click.option(
    '--cap_info_nce_layers',
    default=2,
    type=int,
    help='Number of layers in lang_sup_criterion')
def main(**kwargs):
    exp_base_dir = coco_paths['exp_dir']
    if kwargs['dataset']=='flickr':
        exp_base_dir = flickr_paths['exp_dir']
    exp_const = ExpConstants(kwargs['exp_name'],exp_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.seed = 0
    exp_const.contextualize = not kwargs['no_context']
    exp_const.random_lang = kwargs['random_lang']

    data_const = FlickrDatasetConstants(kwargs['subset'])

    model_const = Constants()
    model_const.model_num = kwargs['model_num']
    model_const.object_encoder = ObjectEncoderConstants()
    model_const.object_encoder.context_layer.output_attentions = True
    model_const.object_encoder.object_feature_dim = 2048
    model_const.cap_encoder = CapEncoderConstants()
    model_const.cap_encoder.output_attentions = True
    model_const.cap_info_nce_layers = kwargs['cap_info_nce_layers']
    if model_const.model_num==-100:
        filename = os.path.join(
            exp_const.exp_dir,
            f'results_val_best.json')
        results = io.load_json_object(filename)
        model_const.model_num = results['model_num']
        print('Selected model num:',model_const.model_num)

    model_const.object_encoder_path = os.path.join(
        exp_const.model_dir,
        f'object_encoder_{model_const.model_num}')
    model_const.lang_sup_criterion_path = os.path.join(
        exp_const.model_dir,
        f'lang_sup_criterion_{model_const.model_num}')
    if exp_const.random_lang is True:
        model_const.cap_encoder_path = os.path.join(
            exp_const.model_dir,
            f'cap_encoder_{model_const.model_num}')

    eval_flickr_phrase_loc.main(exp_const,data_const,model_const)


if __name__=='__main__':
    main()