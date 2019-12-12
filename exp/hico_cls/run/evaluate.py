import os
import click

from utils.constants import Constants, ExpConstants
from global_constants import hico_paths
from ..dataset import HICOFeatDatasetConstants
from ..models.object_encoder import ObjectEncoderConstants
from ..models.hoi_classifier import HOIClassifierConstants
from ..evaluate import main as evaluate


@click.command()
@click.option(
    '--exp_name',
    default='default_exp',
    help='Experiment name')
@click.option(
    '--exp_base_dir',
    default=hico_paths['exp_dir'],
    help='Output directory where a folder would be created for each experiment')
@click.option(
    '--model_num',
    default=-1,
    type=int,
    help='Model number. -1 implies begining of training.')
@click.option(
    '--skip_object_context_layer',
    is_flag=True,
    help='Set flag to skip context layer in object encoder')
def main(**kwargs):
    exp_const = ExpConstants(kwargs['exp_name'],kwargs['exp_base_dir'])
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'logs')
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
    exp_const.batch_size = 50
    exp_const.num_workers = 10
    exp_const.seed = 0
    exp_const.skip_object_context_layer = kwargs['skip_object_context_layer']

    data_const = HICOFeatDatasetConstants('test')

    model_const = Constants()
    model_const.model_num = kwargs['model_num']
    model_const.object_encoder = ObjectEncoderConstants()
    model_const.object_encoder.context_layer.output_attentions = False
    model_const.object_encoder.skip_context_layer = \
        exp_const.skip_object_context_layer
    model_const.hoi_classifier = HOIClassifierConstants()
    model_const.hoi_classifier.context_layer.output_attentions = False
    
    if model_const.model_num==-100:
        model_const.object_encoder_path = os.path.join(
            exp_const.model_dir,
            f'best_object_encoder')
        model_const.hoi_classifier_path = os.path.join(
            exp_const.model_dir,
            f'best_hoi_classifier')
    else:
        model_const.object_encoder_path = os.path.join(
            exp_const.model_dir,
            f'object_encoder_{model_const.model_num}')
        model_const.hoi_classifier_path = os.path.join(
            exp_const.model_dir,
            f'hoi_classifier_{model_const.model_num}')

    print('Evaluating model:', exp_const.exp_name)
    evaluate(exp_const,data_const,model_const)


if __name__=='__main__':
    main()