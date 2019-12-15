import os
import click
from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import utils.io as io
from global_constants import hico_paths
from .conf_mat_utils import ConfMatAggregator


def save_conf_mat_plot(mat,labels,filename):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(mat, interpolation='nearest')
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)

    plt.xticks(rotation=90)
    plt.savefig(filename,bbox_inches='tight')

    plt.close()


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
    '--subset',
    default='test',
    help='Subset to visualize confusion matrices for')
def main(**kwargs):
    exp_dir = os.path.join(kwargs['exp_base_dir'],kwargs['exp_name'])
    subset = kwargs['subset']
    probs_npy = os.path.join(exp_dir,f'prob_{subset}.npy')
    probs = np.load(probs_npy)
    max_probs = np.max(probs,0,keepdims=True)
    min_probs = np.min(probs,0,keepdims=True)
    probs = (probs - min_probs) / (max_probs - min_probs)
    
    pos_labels_npy = os.path.join(exp_dir,f'pos_labels_{subset}.npy')
    pos_labels = np.load(pos_labels_npy)

    conf_mat_agg = ConfMatAggregator()
    num_samples = pos_labels.shape[0]

    for i in tqdm(range(num_samples)):
        conf_mat_agg.update(pos_labels[i],probs[i])

    conf_mat = conf_mat_agg.conf_mat

    print('Creating interaction confusion matrices ...')
    for object_name in tqdm(conf_mat_agg.list_of_objects()):
        mat, mat_labels = \
            conf_mat_agg.conf_mat_object(object_name)
        
        vis_dir = os.path.join(
            exp_dir,
            f'vis/conf_mat/{subset}/interaction_confusion')
        io.mkdir_if_not_exists(vis_dir,recursive=True)

        filename = os.path.join(vis_dir,f'{object_name}.png')
        save_conf_mat_plot(mat,mat_labels,filename)

    
    print('Creating object confusion matrices ...')
    for interaction_name in tqdm(conf_mat_agg.list_of_interactions()):
        mat, mat_labels = \
            conf_mat_agg.conf_mat_interaction(interaction_name)
        
        vis_dir = os.path.join(
            exp_dir,
            f'vis/conf_mat/{subset}/object_confusion')
        io.mkdir_if_not_exists(vis_dir,recursive=True)

        filename = os.path.join(vis_dir,f'{interaction_name}.png')
        save_conf_mat_plot(mat,mat_labels,filename)
    

if __name__=='__main__':
    main()
