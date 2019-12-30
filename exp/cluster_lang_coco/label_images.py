import os
import click
from tqdm import tqdm

import utils.io as io

@click.command()
@click.option(
    '--feat_info_json',
    type=str,
    help='Path to feature info json')
@click.option(
    '--active_words_json',
    type=str,
    help='Path to active words json')
@click.option(
    '--image_labels_json',
    type=str,
    help='Path to image labels json')
@click.option(
    '--labels_json',
    type=str,
    help='Path to labels json')
def main(**kwargs):
    feat_info = io.load_json_object(kwargs['feat_info_json'])
    active_words = io.load_json_object(kwargs['active_words_json'])
    
    labels = []
    all_active_words = set()
    for cluster_id, words in enumerate(active_words):
        all_active_words.update(set(words))
        for word in words:
            labels.append(f'{word}_{cluster_id}')
    
    for word in all_active_words:
        labels.append(f'{word}_-1')

    labels = sorted(labels)
    io.dump_json_object(labels,kwargs['labels_json'])

    image_labels = {}
    for word_info in tqdm(feat_info):
        image_id = word_info['image_id']
        cluster_id = word_info['cluster_id']
        word = word_info['word']

        if word in active_words[cluster_id]:
            label = f'{word}_{cluster_id}'
        elif word in all_active_words:
            label = f'{word}_-1'
        else:
            continue

        if image_id not in image_labels:
            image_labels[image_id] = set()
        
        image_labels[image_id].add(label)

    for k,v in image_labels.items():
        image_labels[k] = list(v)
        
    io.dump_json_object(image_labels,kwargs['image_labels_json'])


if __name__=='__main__':
    main()