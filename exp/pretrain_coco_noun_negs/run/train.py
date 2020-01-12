import os
import click

from utils.constants import Constants, ExpConstants
from global_constants import coco_paths
from ..dataset import DetFeatDatasetConstants
from ..self_sup_dataset import SelfSupDetFeatDatasetConstants
from ..models.object_encoder import ObjectEncoderConstants
from ..models.cap_encoder import CapEncoderConstants
from ..train import main as train


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
    '--neg_noun_loss_wt',
    default=1.0,
    type=float,
    help='Weight for negative verb loss')
@click.option(
    '--self_sup_loss_wt',
    default=0.0,
    type=float,
    help='Weight for self supervision loss')
@click.option(
    '--lang_sup_loss_wt',
    default=1.0,
    type=float,
    help='Weight for language supervision loss')
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
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'logs')
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
    exp_const.optimizer = 'Adam'
    exp_const.lr = kwargs['lr']
    exp_const.momentum = None
    exp_const.num_epochs = 10
    exp_const.log_step = 20
    exp_const.model_save_step = 8000 # 400000/50
    exp_const.val_step = 4000
    exp_const.num_val_samples = None
    exp_const.train_batch_size = kwargs['train_batch_size']
    exp_const.val_batch_size = 20
    exp_const.num_workers = 10
    exp_const.seed = 0
    exp_const.neg_noun_loss_wt = kwargs['neg_noun_loss_wt']
    exp_const.self_sup_loss_wt = kwargs['self_sup_loss_wt']
    exp_const.lang_sup_loss_wt = kwargs['lang_sup_loss_wt']
    exp_const.contextualize = not kwargs['no_context']
    exp_const.self_sup_feat = kwargs['self_sup_feat']
    
    DatasetConstants = DetFeatDatasetConstants
    if exp_const.self_sup_feat==True:
        DatasetConstants = SelfSupDetFeatDatasetConstants
    
    data_const = {
        'train': DatasetConstants('train'),
        'val': DatasetConstants('val'),
    }

    model_const = Constants()
    model_const.model_num = kwargs['model_num']
    model_const.object_encoder = ObjectEncoderConstants()
    model_const.object_encoder.context_layer.output_attentions = True
    if exp_const.self_sup_feat==True:
        model_const.object_encoder.object_feature_dim = 1024 + 256
    model_const.cap_encoder = CapEncoderConstants()
    model_const.cap_encoder.output_attentions = True
    model_const.object_encoder_path = os.path.join(
        exp_const.model_dir,
        f'object_encoder_{model_const.model_num}')
    model_const.self_sup_criterion_path = os.path.join(
        exp_const.model_dir,
        f'self_sup_criterion_{model_const.model_num}')
    model_const.lang_sup_criterion_path = os.path.join(
        exp_const.model_dir,
        f'lang_sup_criterion_{model_const.model_num}')

    train(exp_const,data_const,model_const)


if __name__=='__main__':
    main()