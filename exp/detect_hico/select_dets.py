import os
import h5py
import glob
import click
import numpy as np
from tqdm import tqdm

import utils.io as io
from .coco_classes import det_cls_id_to_coco_cls_id, HICO_COCO_CLASSES

def get_cls_id_to_rpn_ids(labels):
    num_boxes = len(labels)
    rpn_ids = np.arange(num_boxes)
    cls_id_to_rpn_ids = [None]*len(HICO_COCO_CLASSES)
    for i in range(len(cls_id_to_rpn_ids)):
        cls_id_to_rpn_ids[i] = set()

    for rpn_id, det_cls_id in enumerate(labels):
        coco_cls_id = det_cls_id_to_coco_cls_id[det_cls_id]
        cls_id_to_rpn_ids[coco_cls_id].add(rpn_id)
    
    return cls_id_to_rpn_ids


@click.command()
@click.option(
    '--det_dir',
    type=str,
    help='Detection directory')
def main(**kwargs):
    f = {}
    for f_type in ['scores','boxes','labels']:
        f[f_type] = io.load_h5py_object(
            os.path.join(kwargs['det_dir'],f'{f_type}.hdf5'))

    filename = os.path.join(kwargs['det_dir'],'selected_coco_cls_dets.hdf5')
    selected_f = h5py.File(filename,'w')

    for global_id in tqdm(f['boxes'].keys()):
        boxes = f['boxes'][global_id][()]
        scores = f['scores'][global_id][()]
        labels = f['labels'][global_id][()]
        cls_id_to_rpn_ids = get_cls_id_to_rpn_ids(labels)
        boxes_scores_rpn_ids = np.zeros([len(boxes),6],dtype=np.float32)
        start_end_ids = np.zeros([len(HICO_COCO_CLASSES),2],dtype=np.int32)
        start_id = 0
        for cls_id, rpn_ids in enumerate(cls_id_to_rpn_ids):
            for i, rpn_id in enumerate(rpn_ids):
                k = start_id + i
                boxes_scores_rpn_ids[k,:4] = boxes[rpn_id]
                boxes_scores_rpn_ids[k,4] = scores[rpn_id]
                boxes_scores_rpn_ids[k,5] = rpn_id

            start_end_ids[cls_id,0] = start_id
            start_end_ids[cls_id,1] = start_id + len(rpn_ids)
            start_id = start_id + len(rpn_ids)

        selected_f.create_group(global_id)
        selected_f[global_id].create_dataset(
            'boxes_scores_rpn_ids',
            data=boxes_scores_rpn_ids)
        selected_f[global_id].create_dataset(
            'start_end_ids',
            data=start_end_ids)




if __name__=='__main__':
    main()
