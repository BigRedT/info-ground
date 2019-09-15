import os
import h5py
import click
import random
import shutil
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import NearestNeighbors
import skimage.io as skio

import utils.io as io
from utils.bbox_utils import vis_bbox
from utils.html_writer import HtmlWriter
from detector.model import COCO_INSTANCE_CATEGORY_NAMES
from global_constants import coco_paths
from data.coco.constants import CocoConstants


def select_queries(object_index,num_queries):
    num_objects = len(object_index)
    queries = [None]*num_objects
    for i in range(num_objects):
        if len(object_index[i]) <= num_queries:
            queries[i] = []
            continue

        queries[i] = random.sample(object_index[i],num_queries)
    
    return queries


def create_search_index(object_index,queries,features_f,search_features_f):
    num_objects = len(object_index)
    search_objects = [None]*num_objects
    for i in tqdm(range(num_objects)):
        #print(f'{i}/{num_objects}')
        search_objects_ = []
        search_features_ = []
        object_queries = queries[i]
        query_image_ids = set([q[0] for q in object_queries])
        object_index_ = random.sample(
            object_index[i],
            min(len(object_index[i]),5000))
        for image_id_obj_id in object_index_:
            if image_id_obj_id[0] in query_image_ids:
                continue
                
            search_objects_.append(image_id_obj_id)

            image_id, obj_id = image_id_obj_id
            search_features_.append(features_f[image_id][obj_id][()])
        
        if len(search_objects_)==0:
            continue

        search_objects[i] = search_objects_
        search_features_ = np.stack(search_features_)
        search_features_f.create_dataset(str(i),data=search_features_)
    
    return search_objects, search_features_f


def get_query_features(queries,features_f):
    num_objects = len(queries)
    query_features = [None]*num_objects
    for i in range(num_objects):
        query_features_ = []
        for image_id,obj_id in queries[i]:
            query_features_.append(features_f[image_id][obj_id][()])

        if len(query_features_) > 0:
            query_features_ = np.stack(query_features_)
        
        query_features[i] = query_features_
    
    return query_features


def find_nn(query_features,search_features_f,num_nbrs):
    num_objects = len(query_features)
    nbr_ids = [None]*num_objects
    for i in tqdm(range(num_objects)):
        if len(query_features[i])==0:
            nbr_ids[i] = []
            continue
        
        search_features_ = search_features_f[str(i)][()]
        nbrs = NearestNeighbors(
            n_neighbors=num_nbrs,
            algorithm='ball_tree').fit(search_features_)

        _, nbr_ids_ = nbrs.kneighbors(query_features[i])
        nbr_ids[i] = nbr_ids_
    
    return nbr_ids


def get_image_id_to_cap(subset):
    coco_const = CocoConstants(coco_paths)
    annos = io.load_json_object(coco_const.caption_annos_json[subset])
    annos = annos['annotations']
    image_id_to_cap = {}
    for anno in tqdm(annos):
        image_id = 'COCO_' + \
            coco_paths['extracted']['images'][subset] + '_' + \
            str(anno['image_id']).zfill(12)
        caption = anno['caption']
        if image_id not in image_id_to_cap:
            image_id_to_cap[image_id] = []
        
        image_id_to_cap[image_id].append(caption)
    
    return image_id_to_cap



def write_html(knn,vis_dir,image_dir,boxes_hdf5):
    io.mkdir_if_not_exists(vis_dir,recursive=True)
    num_objects = len(knn)

    boxes_f = io.load_h5py_object(boxes_hdf5)
    for i in tqdm(range(num_objects)):
        obj_name = COCO_INSTANCE_CATEGORY_NAMES[i]
        if obj_name=='N/A':
            continue
        
        obj_dir = os.path.join(vis_dir,obj_name)
        io.mkdir_if_not_exists(obj_dir)
        
        filename = os.path.join(obj_dir,'vis.html')
        html_writer = HtmlWriter(filename)
        
        col_dict = {0:'Query'}
        html_writer.add_element(col_dict)

        num_queries = len(knn[i])
        for j in range(num_queries):
            if knn[i][j] is None:
                continue

            query_image_id, query_box_id = knn[i][j]['query_object']
            src_filename = os.path.join(image_dir,query_image_id+'.jpg')
            dst_filename = os.path.join(
                obj_dir,
                f'{query_image_id}_{query_box_id}.jpg')
            image = skio.imread(src_filename)
            bbox = boxes_f[query_image_id][query_box_id]
            vis_bbox(bbox,image,modify=True)
            skio.imsave(dst_filename,image)
            col_dict = {0: html_writer.image_tag(
                f'{query_image_id}_{query_box_id}.jpg')}

            num_retrieved = len(knn[i][j]['retrieved_caps'])
            for k in range(num_retrieved):
                retrieved_image_id, retrieved_box_id = \
                    knn[i][j]['retrieved_objects'][k]
                src_filename = os.path.join(image_dir,retrieved_image_id+'.jpg')
                dst_filename = os.path.join(
                    obj_dir,
                    f'{retrieved_image_id}_{retrieved_box_id}.jpg')
                image = skio.imread(src_filename)
                bbox = boxes_f[retrieved_image_id][retrieved_box_id]
                vis_bbox(bbox,image,modify=True)
                skio.imsave(dst_filename,image)
                col_dict[1+k] = html_writer.image_tag(
                    f'{retrieved_image_id}_{retrieved_box_id}.jpg')
            
            html_writer.add_element(col_dict)

            col_dict = {0: knn[i][j]['query_cap']}
            num_retrieved = len(knn[i][j]['retrieved_caps'])
            for k in range(num_retrieved):
                col_dict[1+k] = knn[i][j]['retrieved_caps'][k]

            html_writer.add_element(col_dict)

            col_dict = {0: knn[i][j]['query_object']}
            num_retrieved = len(knn[i][j]['retrieved_objects'])
            for k in range(num_retrieved):
                col_dict[1+k] = knn[i][j]['retrieved_objects'][k]
            
            html_writer.add_element(col_dict)

        html_writer.close()
    
    boxes_f.close()
        

@click.command()
@click.option(
    '--object_index_json',
    type=str,
    help='Path to object index json relative to proc_dir')
@click.option(
    '--boxes_hdf5',
    type=str,
    help='Path to detection boxes hdf5')
@click.option(
    '--features_hdf5',
    type=str,
    help='Path to the features hdf5 file')
@click.option(
    '--num_queries',
    type=int,
    help='Number of queries per object category')
@click.option(
    '--search_objects_json',
    type=str,
    help='Path to search objects json')
@click.option(
    '--search_features_hdf5',
    type=str,
    help='Path to search features hdf5')
@click.option(
    '--num_nbrs',
    type=int,
    help='Number of nbrs to retrieve')
@click.option(
    '--anno_subset',
    type=str,
    help='Subset to use for loading annotations')
@click.option(
    '--knn_json',
    type=str,
    help='Path to the output json file with retreived captions')
@click.option(
    '--vis_dir',
    type=str,
    help='Path to visualization directory where html files would be saved')
@click.option(
    '--load_knn_json',
    is_flag=True,
    help='Loads an existing knn json file')
def main(**kwargs):
    random.seed(0)
    os.environ['HDF5_USE_FILE_LOCKING']="FALSE"

    if kwargs['load_knn_json'] is not True:
        features_f = io.load_h5py_object(kwargs['features_hdf5'])
        object_index = io.load_json_object(kwargs['object_index_json'])[:20]
        num_objects = len(object_index)

        print('Creating image_id to caption map ...')
        image_id_to_cap = get_image_id_to_cap(kwargs['anno_subset'])

        print('Selecting queries ...')
        queries = select_queries(object_index,kwargs['num_queries'])
        query_features = get_query_features(queries,features_f)

        print('Creating search index ...')
        search_features_f = h5py.File(kwargs['search_features_hdf5'],'w')
        search_object, search_features_f = create_search_index(
            object_index,
            queries,
            features_f,
            search_features_f)

        print('Find nearest neighbors ...')
        nbr_ids = find_nn(query_features,search_features_f,kwargs['num_nbrs'])
        nbrs = [None]*num_objects
        for i in range(num_objects):
            nbrs_ = [None]*kwargs['num_queries']
            search_object_ = search_object[i]
            nbr_ids_ = nbr_ids[i]
            num_queries_ = len(nbr_ids_)
            for j in range(num_queries_):
                retrieved_nbr_ids = nbr_ids_[j]
                num_nbrs_ = len(retrieved_nbr_ids)
                retrieved_nbrs = [None]*num_nbrs_
                for k in range(num_nbrs_):
                    retrieved_image_id = retrieved_nbr_ids[k]
                    retrieved_nbrs[k] = search_object_[retrieved_image_id]
                
                nbrs_[j] = retrieved_nbrs
            
            nbrs[i] = nbrs_
        
        print('Get captions ...')
        knn = [None]*num_objects
        for i in range(num_objects):
            knn_ = [None]*kwargs['num_queries']
            nbrs_ = nbrs[i]
            queries_ = queries[i]
            num_queries_ = len(queries_)
            for j in range(num_queries_):
                query_image_id = queries_[j][0]
                query_cap = image_id_to_cap[query_image_id]
                retrieved_nbrs = nbrs_[j]
                num_nbrs_ = len(retrieved_nbrs)
                retrieved_caps = [None]*num_nbrs_
                for k in range(num_nbrs_):
                    retrieved_image_id = retrieved_nbrs[k][0]
                    retrieved_caps[k] = image_id_to_cap[retrieved_image_id]
                
                knn_[j] = {
                    'query_cap': query_cap,
                    'retrieved_caps': retrieved_caps,
                    'query_object': queries_[j],
                    'retrieved_objects': retrieved_nbrs}
            
            knn[i] = knn_

        io.dump_json_object(knn,kwargs['knn_json'])

        features_f.close()
        search_features_f.close()

    else:
        print('Loading existing knn json ...')
        knn = io.load_json_object(kwargs['knn_json'])

    subset = kwargs['anno_subset']
    image_dir = os.path.join(
        coco_paths['image_dir'],
        coco_paths['extracted']['images'][subset])

    write_html(knn,kwargs['vis_dir'],image_dir,kwargs['boxes_hdf5'])


if __name__=='__main__':
    main()
