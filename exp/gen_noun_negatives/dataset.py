import os
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import random
import utils.io as io
from utils.constants import Constants
from global_constants import coco_paths


class DetFeatDatasetConstants(Constants):
    def __init__(self,subset):
        super().__init__()
        self.subset = subset
        self.subset_image_dirname = coco_paths['extracted']['images'][subset]
        # self.det_dir = os.path.join(
        #     coco_paths['local_proc_dir'],
        #     f'detections/{subset}')
        # self.det_dir = os.path.join(
        #     coco_paths['proc_dir'],
        #     f'detections/{subset}')
        # self.boxes_hdf5 = os.path.join(self.det_dir,'boxes.hdf5')
        # self.features_hdf5 = os.path.join(self.det_dir,'features.hdf5')
        # self.labels_hdf5 = os.path.join(self.det_dir,'labels.hdf5')
        # self.scores_hdf5 = os.path.join(self.det_dir,'scores.hdf5')
        self.annos_json = os.path.join(
            coco_paths['proc_dir'],
            coco_paths['extracted']['annos']['captions'][subset])
        self.noun_tokens_json = os.path.join(
            coco_paths['proc_dir'],
            coco_paths['extracted']['annos']['noun_tokens'][subset])
        self.noun_vocab_json = os.path.join(
            coco_paths['proc_dir'],
            coco_paths['extracted']['annos']['noun_vocab'][subset])


class DetFeatDataset(Dataset):
    def __init__(self,const):
        self.const = deepcopy(const)
        self.annos = io.load_json_object(self.const.annos_json)
        self.noun_token_ids = io.load_json_object(
            self.const.noun_tokens_json)
        os.environ['HDF5_USE_FILE_LOCKING']="FALSE"
        
    def get_image_name(self,subset,image_id):
        image_id = str(image_id).zfill(12)
        return f'COCO_{self.const.subset_image_dirname}_{image_id}'
        
    def __len__(self):
        return len(self.annos['annotations'])

    def __getitem__(self, i):
        anno = self.annos['annotations'][i]
        image_id = anno['image_id']
        cap_id = anno['id']
        caption = anno['caption']
        image_name = self.get_image_name(self.const.subset,anno['image_id'])
        to_return = {
            'image_id': image_id,
            'cap_id': cap_id,
            'image_name': image_name,
            'caption': caption,
        }
        
        noun_token_ids = self.noun_token_ids[i]['token_ids']
        if len(noun_token_ids)==0:
            to_return['noun_token_ids'] = []
        else:
            select_noun_token_ids = random.choice(noun_token_ids)
            to_return['noun_token_ids'] = \
                np.array(select_noun_token_ids,dtype=np.int32)

        return to_return

    def get_collate_fn(self):
        def collate_fn(batch):
            new_batch = {}
            for k in batch[0].keys():
                batch_k = [sample[k] for sample in batch]
                if k=='noun_token_ids':
                    new_batch[k] = batch_k
                else:
                    new_batch[k] = default_collate(batch_k)
            
            return new_batch
        
        return collate_fn

if __name__=='__main__':
    const = DetFeatDatasetConstants('val')
    dataset = DetFeatDataset(const)
    collate_fn = dataset.get_collate_fn()
    dataloader = DataLoader(dataset,5,num_workers=3,collate_fn=collate_fn)
    for data in dataloader:
        import pdb; pdb.set_trace()
