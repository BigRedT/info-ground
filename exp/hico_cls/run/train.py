import os
import click

from utils.constants import Constants, ExpConstants
from global_constants import hico_paths
from ..dataset import HICOFeatDatasetConstants
from ..models.object_encoder import ObjectEncoderConstants
from ..models.hoi_classifier import HOIClassifierConstants
from ..train import main as train


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
    '--lr',
    default=1e-5,
    type=float,
    help='Learning rate')
@click.option(
    '--train_batch_size',
    default=200,
    type=int,
    help='Training batch size')
@click.option(
    '--finetune_object_encoder',
    is_flag=True,
    help='Set this flag to finetune object encoder')
@click.option(
    '--skip_object_context_layer',
    is_flag=True,
    help='Set flag to skip context layer in object encoder')
@click.option(
    '--pretrained_object_encoder_path',
    default='unavailable',
    help='Set to the path of the pretrained encoder')
@click.option(
    '--warmup',
    is_flag=True,
    help='Would freeze the context layers for warmup (first 20 epochs)')
@click.option(
    '--ignore_unk_labels_during_training',
    is_flag=True,
    help='Set flag to only train on positive and negative labels')
@click.option(
    '--balanced_bce',
    is_flag=True,
    help='Set flag to train using a balanced BCE loss')
def main(**kwargs):
    exp_const = ExpConstants(kwargs['exp_name'],kwargs['exp_base_dir'])
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'logs')
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
    exp_const.optimizer = 'Adam'
    exp_const.lr = kwargs['lr']
    exp_const.momentum = None
    exp_const.num_epochs = 50
    exp_const.log_step = 20
    exp_const.model_save_step = 1000 # 30000/50
    exp_const.val_step = 1000
    exp_const.num_val_samples = None
    exp_const.train_batch_size = kwargs['train_batch_size']
    exp_const.val_batch_size = 50
    exp_const.num_workers = 10
    exp_const.seed = 0
    exp_const.finetune_object_encoder = kwargs['finetune_object_encoder']
    exp_const.skip_object_context_layer = kwargs['skip_object_context_layer']
    exp_const.pretrained_object_encoder_path = \
        kwargs['pretrained_object_encoder_path']
    exp_const.warmup = kwargs['warmup']
    exp_const.ignore_unk_labels_during_training = \
        kwargs['ignore_unk_labels_during_training']
    exp_const.balanced_bce = kwargs['balanced_bce']
    
    data_const = {
        'train': HICOFeatDatasetConstants('train'),
        'val': HICOFeatDatasetConstants('val'),
    }

    model_const = Constants()
    model_const.model_num = kwargs['model_num']
    model_const.object_encoder = ObjectEncoderConstants()
    model_const.object_encoder.context_layer.output_attentions = False
    model_const.object_encoder.skip_context_layer = \
        exp_const.skip_object_context_layer
    model_const.hoi_classifier = HOIClassifierConstants()
    model_const.hoi_classifier.context_layer.output_attentions = False
    model_const.object_encoder_path = os.path.join(
        exp_const.model_dir,
        f'object_encoder_{model_const.model_num}')
    model_const.hoi_classifier_path = os.path.join(
        exp_const.model_dir,
        f'hoi_classifier_{model_const.model_num}')

    train(exp_const,data_const,model_const)


if __name__=='__main__':
    main()