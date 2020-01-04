import os
import click
import numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import Dataset, DataLoader

import utils.io as io
from global_constants import coco_paths
from .dataset import ClusterDatasetConstants, ClusterDataset


def cluster(dataloader, n_clusters, batch_size, exp_dir):
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=0,
        batch_size=batch_size)

    for epoch in range(20):
        inertia = 0
        for data in tqdm(dataloader):
            feat = data['feat'].detach().numpy()
            kmeans.partial_fit(feat)
            inertia += kmeans.inertia_

        print('epoch:',epoch,'inertia:',inertia)
            
        print('Saving cluster centers ...')
        cluster_centers_npy = os.path.join(
            exp_dir,
            'cluster_centers.npy')
        np.save(cluster_centers_npy,kmeans.cluster_centers_)

    return kmeans.cluster_centers_


def pred_cluster_ids(dataloader, cluster_centers, exp_dir):
    n_clusters = cluster_centers.shape[0]
    
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=0,
        batch_size=dataloader.batch_size)
    kmeans.cluster_centers_ = cluster_centers


    feat_info = [info for idx, info in dataloader.dataset.filtered_info]
    clustered_sentences = [None]*n_clusters
    for data in tqdm(dataloader):
        pred_clusters = kmeans.predict(data['feat'].detach().numpy())
        feat_info_idx = data['filtered_info_idx'].detach().numpy()
        for i,cap in enumerate(data['caption']):
            k = pred_clusters[i]
            if clustered_sentences[k] is None:
                clustered_sentences[k] = []
                
            clustered_sentences[k].append((data['word'][i],cap))
            feat_info[feat_info_idx[i]]['cluster_id'] = k

    print('Saving clustered sentences ...')
    io.dump_json_object(
        clustered_sentences,
        os.path.join(exp_dir,'clustered_sentences.json'))
    
    print('Saving feat info with predicted cluster ids ...')
    io.dump_json_object(
        feat_info,
        os.path.join(exp_dir,'feat_info.json'))

    return clustered_sentences, feat_info


def compute_activity(clustered_sentences, exp_dir):
    num_active_clusters = 0
    active_nouns = [None]*len(clustered_sentences)
    for i,cluster in enumerate(clustered_sentences):
        counts = {}

        if cluster is not None:
            for word, sentence in cluster:
                if word not in counts:
                    counts[word] = 0
                
                counts[word] += 1

        active_nouns[i] = set()
        for word, count in counts.items():
            if count > 250:
                active_nouns[i].add(word)

        active_nouns[i] = list(active_nouns[i])
        num_active_clusters += len(active_nouns[i])
    
    print('Saving active nouns ...')
    io.dump_json_object(
        active_nouns,
        os.path.join(exp_dir,'active_nouns.json'))
    
    print('Num active clusters:',num_active_clusters)
    
    return active_nouns, num_active_clusters


def save_clusters(clustered_sentences, active_nouns, exp_dir):
    cluster_dir = os.path.join(exp_dir,'clustered_sentences')
    io.mkdir_if_not_exists(cluster_dir)
    n_clusters = len(active_nouns)
    for i in tqdm(range(n_clusters)):
        for active_word in active_nouns[i]:
            sentences = [sen for word, sen in clustered_sentences[i] \
                if word==active_word]
            filename = os.path.join(cluster_dir,f'{active_word}_{i}.json')
            io.dump_json_object(sentences,filename)



@click.command()
@click.option(
    '--batch_size',
    default=2000,
    type=int,
    help='Batch size')
@click.option(
    '--n_clusters',
    default=1000,
    type=int,
    help='Number of clusters for kmeans')
@click.option(
    '--filter_words',
    default='_everything_',
    type=str,
    help='word to filter data by')
@click.option(
    '--fit',
    is_flag=True,
    help='set flag to fit kmeans')
@click.option(
    '--predict',
    is_flag=True,
    help='set flag to predict')
def main(**kwargs):
    exp_dir = os.path.join(
        coco_paths['exp_dir'],
        'bert_noun_clusters_' + str(kwargs['n_clusters']))
    io.mkdir_if_not_exists(exp_dir)    

    data_const = ClusterDatasetConstants('train')
    if kwargs['filter_words']=='_everything_':
        data_const.filter_words = None
    else:
        filter_words = kwargs['filter_words'].split('_') 
        data_const.filter_words = set(filter_words)

    dataset = ClusterDataset(data_const)
    dataloader = DataLoader(
        dataset,
        batch_size=kwargs['batch_size'],
        shuffle=True,
        num_workers=60)
    print('Num samples:', len(dataset))

    if kwargs['fit']==True:
        cluster_centers = cluster(dataloader, kwargs['n_clusters'],kwargs['batch_size'], exp_dir)
    else:
        cluster_centers_npy = os.path.join(
            exp_dir,
            'cluster_centers.npy')
        cluster_centers = np.load(cluster_centers_npy)

    if kwargs['predict']==True:
        clustered_sentences, feat_info = pred_cluster_ids(
            dataloader, 
            cluster_centers, 
            exp_dir)
        active_nouns, num_active_clusters = compute_activity(
            clustered_sentences, 
            exp_dir)
        save_clusters(clustered_sentences, active_nouns, exp_dir)


if __name__=='__main__':
    main()