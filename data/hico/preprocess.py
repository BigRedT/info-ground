import os
import click
import random
import numpy as np
import scipy.io as scio

import utils.io as io
from global_constants import hico_paths


def get_hois(anno):
    list_action = anno['list_action']
    num_hois = len(list_action)
    hois = [None]*num_hois
    for i in range(num_hois):
        hois[i] = {
            'id': i,
            'object': list_action[i][0][0][0],
            'interaction': list_action[i][0][1][0]
        }
    
    return hois
    

def get_subset_list(anno,key='train'):
    list_subset = anno[f'list_{key}']
    return [sample[0][0] for sample in list_subset]


def get_train_val_ids(num_train_val_samples,split=0.7):
    ids = list(range(num_train_val_samples))
    random.shuffle(ids)
    num_train_samples = int(num_train_val_samples*split)
    train_ids = ids[:num_train_samples]
    val_ids = ids[num_train_samples:]
    return train_ids, val_ids


def get_labels(anno,key='train'):
    return anno[f'anno_{key}'].transpose()


def main():
    random.seed(0)
    
    print('Reading anno.mat ...')
    anno_mat = os.path.join(hico_paths['downloads_dir'],hico_paths['anno_mat'])
    anno = scio.loadmat(anno_mat)

    print('Getting list of hois ...')
    hoi_list = get_hois(anno)
    filename = os.path.join(hico_paths['proc_dir'],hico_paths['hoi_list_json'])
    io.dump_json_object(hoi_list,filename)
    
    print('Getting subset lists ...')
    train_val_list = get_subset_list(anno,key='train')
    test_list = get_subset_list(anno,key='test')

    print('Splitting train_val_list into train_list and val_list ...')
    num_train_val_samples = len(train_val_list)
    train_ids, val_ids = get_train_val_ids(num_train_val_samples,split=0.8)
    train_list = [train_val_list[i] for i in train_ids]
    val_list = [train_val_list[i] for i in val_ids]

    print('Getting labels ...')
    train_val_labels = get_labels(anno,key='train')
    test_labels = get_labels(anno,key='test')

    print('Splitting train_val_labels in train_labels and val_labels ...')
    train_labels = train_val_labels[train_ids]
    val_labels = train_val_labels[val_ids]

    print('Saving subset_lists and subset_labels ...')
    for subset in ['train','val','test','train_val']:
        filename = os.path.join(
            hico_paths['proc_dir'],hico_paths['subset_list_json'][subset])
        io.dump_json_object(locals()[f'{subset}_list'],filename)

        filename = os.path.join(
            hico_paths['proc_dir'],hico_paths['labels_npy'][subset])
        np.save(filename,locals()[f'{subset}_labels'])


if __name__=='__main__':
    main()