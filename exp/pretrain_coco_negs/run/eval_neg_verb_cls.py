import os
import click

from utils.constants import Constants, ExpConstants
from global_constants import coco_paths
from ..dataset import DetFeatDatasetConstants
from ..models.object_encoder import ObjectEncoderConstants
from ..models.cap_encoder import CapEncoderConstants
from ..eval_neg_verb_cls import main as verb_cls


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
    exp_const.num_val_samples = None
    exp_const.batch_size = 50
    exp_const.num_workers = 10
    exp_const.seed = 0

    data_const = DetFeatDatasetConstants('val')
    data_const.read_neg_samples = True
    data_const.read_noun_verb_tokens = True

    model_const = Constants()
    model_const.model_num = kwargs['model_num']
    model_const.object_encoder = ObjectEncoderConstants()
    model_const.object_encoder.context_layer.output_attentions = True
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

    verb_cls(exp_const,data_const,model_const)


if __name__=='__main__':
    main()