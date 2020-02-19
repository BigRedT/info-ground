import os
import glob
import click

import utils.io as io
from utils.constants import Constants, ExpConstants
from global_constants import coco_paths
from exp.eval_flickr.dataset import  FlickrDatasetConstants
from exp.eval_flickr.self_sup_dataset import SelfSupFlickrDatasetConstants
from ..models.object_encoder import ObjectEncoderConstants
from ..models.cap_encoder import CapEncoderConstants
from .. import eval_flickr_phrase_loc


def find_all_model_numbers(model_dir):
    model_nums = []
    for name in glob.glob(f'{model_dir}/lang_sup_criterion*'):
        model_nums.append(int(name.split('_')[-1]))

    return sorted(model_nums)


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
    '--no_context',
    is_flag=True,
    help='Apply flag to switch off contextualization')
@click.option(
    '--self_sup_feat',
    is_flag=True,
    help='Apply flag to use self-supervised features')
@click.option(
    '--subset',
    default='val',
    help='subset to run evaluation on')
def main(**kwargs):
    exp_const = ExpConstants(kwargs['exp_name'],kwargs['exp_base_dir'])
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.seed = 0
    exp_const.contextualize = not kwargs['no_context']
    exp_const.self_sup_feat = kwargs['self_sup_feat']

    DatasetConstants = FlickrDatasetConstants
    if exp_const.self_sup_feat==True:
        DatasetConstants = SelfSupFlickrDatasetConstants

    data_const = DatasetConstants(kwargs['subset'])

    model_const = Constants()
    model_const.object_encoder = ObjectEncoderConstants()
    model_const.object_encoder.context_layer.output_attentions = True
    model_const.object_encoder.object_feature_dim = 2048
    if exp_const.self_sup_feat==True:
        model_const.object_encoder.object_feature_dim = 2048 + 256
    model_const.cap_encoder = CapEncoderConstants()
    model_const.cap_encoder.output_attentions = True

    model_nums = find_all_model_numbers(exp_const.model_dir)
    # for num in model_nums:
    #     model_const.model_num = num
    #     model_const.object_encoder_path = os.path.join(
    #         exp_const.model_dir,
    #         f'object_encoder_{model_const.model_num}')
    #     model_const.lang_sup_criterion_path = os.path.join(
    #         exp_const.model_dir,
    #         f'lang_sup_criterion_{model_const.model_num}')
    
    #     eval_flickr_phrase_loc.main(exp_const,data_const,model_const)

    best_model_num = -1
    best_pt_recall = 0
    best_results = None
    for num in model_nums:
        filename = os.path.join(
            exp_const.exp_dir,
            f'results_{data_const.subset}_{num}.json')
        
        if not os.path.exists(filename):
            break

        results = io.load_json_object(filename)
        results['model_num'] = num
        print(results)
        if results['pt_recall'] >= best_pt_recall:
            best_results = results
            best_pt_recall = results['pt_recall']
            best_model_num = num

    print('-'*80)
    best_results['model_num'] = best_model_num
    print(best_results)
    filename = os.path.join(
        exp_const.exp_dir,
        f'results_{data_const.subset}_best.json')
    io.dump_json_object(best_results,filename)


if __name__=='__main__':
    main()