import os
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import utils.io as io
from utils.constants import Constants
from global_constants import coco_paths


class DetFeatDatasetConstants(Constants):
    def __init__(self,subset):
        super().__init__()
        self.subset = subset
        self.subset_image_dirname = coco_paths['extracted']['images'][subset]
        self.det_dir = os.path.join(
            coco_paths['local_proc_dir'],
            f'detections/{subset}')
        # self.det_dir = os.path.join(
        #     coco_paths['proc_dir'],
        #     f'detections/{subset}')
        self.boxes_hdf5 = os.path.join(self.det_dir,'boxes.hdf5')
        self.features_hdf5 = os.path.join(self.det_dir,'features.hdf5')
        self.labels_hdf5 = os.path.join(self.det_dir,'labels.hdf5')
        self.scores_hdf5 = os.path.join(self.det_dir,'scores.hdf5')
        self.annos_json = os.path.join(
            coco_paths['proc_dir'],
            coco_paths['extracted']['annos']['captions'][subset])
        self.max_objects = 15
        self.mask_prob = 0.2
        self.noun_tokens_json = os.path.join(
            coco_paths['proc_dir'],
            coco_paths['extracted']['annos']['noun_tokens'][subset])
        self.read_noun_tokens = False
        self.max_noun_tokens = 6


class DetFeatDataset(Dataset):
    def __init__(self,const):
        self.const = deepcopy(const)
        self.annos = io.load_json_object(self.const.annos_json)
        if self.const.read_noun_tokens is True:
            self.noun_token_ids = io.load_json_object(
                self.const.noun_tokens_json)
        os.environ['HDF5_USE_FILE_LOCKING']="FALSE"
        
    def get_image_name(self,subset,image_id):
        image_id = str(image_id).zfill(12)
        return f'COCO_{self.const.subset_image_dirname}_{image_id}'
        
    def __len__(self):
        return len(self.annos['annotations'])

    def read_object_features(self,image_name):
        f = io.load_h5py_object(self.const.features_hdf5)
        features = f[image_name][()]
        f.close()
        return features
    
    def read_boxes(self,image_name):
        f = io.load_h5py_object(self.const.boxes_hdf5)
        boxes = f[image_name][()]
        f.close()
        return boxes

    def pad_object_features(self,features):
        T,D = features.shape # num_objects x feat. dim
        pad_mask = np.zeros(self.const.max_objects).astype(np.bool)
        if T==self.const.max_objects:
            return features, pad_mask

        if T > self.const.max_objects:
            features = features[:self.const.max_objects]
        else:
            features = np.concatenate((
                features,
                np.zeros([self.const.max_objects-T,D])),0).astype(np.float32)
            pad_mask[T:] = True
            
        return features, pad_mask

    def mask_objects(self,num_objects):
        mask = np.random.uniform(size=(self.const.max_objects))
        mask[num_objects:] = 1.0
        mask = mask < self.const.mask_prob
        return mask

    def pad_noun_token_ids(self,noun_token_ids):
        num_tokens = len(noun_token_ids)
        if num_tokens >= self.const.max_noun_tokens:
            noun_token_ids = noun_token_ids[:self.const.max_noun_tokens]
        else:
            padding = [-1]*(self.const.max_noun_tokens - num_tokens)
            noun_token_ids = noun_token_ids + padding

        return noun_token_ids

    def __getitem__(self, i):
        anno = self.annos['annotations'][i]
        image_id = anno['image_id']
        cap_id = anno['id']
        caption = anno['caption']
        image_name = self.get_image_name(self.const.subset,anno['image_id'])
        features = self.read_object_features(image_name)
        num_objects = features.shape[0]
        features, pad_mask = self.pad_object_features(features)
        object_mask = self.mask_objects(num_objects)
        to_return = {
            'image_id': image_id,
            'cap_id': cap_id,
            'image_name': image_name,
            'caption': caption,
            'features': features,
            'num_objects': num_objects,
            'object_mask': object_mask,
            'pad_mask': pad_mask,
        }
        if self.const.read_noun_tokens is True:
            noun_token_ids = self.noun_token_ids[i]['token_ids']
            to_return['noun_token_ids'] = \
                np.array(self.pad_noun_token_ids(noun_token_ids),dtype=np.int32)

        return to_return


if __name__=='__main__':
    const = DetFeatDatasetConstants('train')
    const.read_noun_tokens = True
    dataset = DetFeatDataset(const)
    print(len(dataset))
    dataloader = DataLoader(dataset,3,num_workers=3)
    for data in dataloader:
        import pdb; pdb.set_trace()
