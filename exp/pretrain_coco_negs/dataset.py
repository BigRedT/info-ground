import os
import numpy as np
import random
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
        self.noun_verb_tokens_json = os.path.join(
            coco_paths['proc_dir'],
            coco_paths['extracted']['annos']['noun_verb_tokens'][subset])
        self.read_noun_verb_tokens = False
        self.max_noun_verb_tokens = 6
        self.neg_samples_json = os.path.join(
            coco_paths['proc_dir'],
            coco_paths['extracted']['negatives']['samples'][subset])
        self.neg_samples_h5py = os.path.join(
            coco_paths['proc_dir'],
            coco_paths['extracted']['negatives']['feats'][subset])
        self.read_neg_samples = False
        self.num_neg_verbs = 5
        self.neg_verb_feat_dim = 768


class DetFeatDataset(Dataset):
    def __init__(self,const):
        self.const = deepcopy(const)
        self.annos = io.load_json_object(self.const.annos_json)
        if self.const.read_noun_verb_tokens is True:
            self.noun_verb_token_ids = io.load_json_object(
                self.const.noun_verb_tokens_json)
        if self.const.read_neg_samples is True:
            self.neg_samples = io.load_json_object(self.const.neg_samples_json)
            self.neg_samples_feats = io.load_h5py_object(
                self.const.neg_samples_h5py)
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

    def pad_noun_verb_token_ids(self,noun_verb_token_ids):
        num_tokens = len(noun_verb_token_ids)
        if num_tokens >= self.const.max_noun_verb_tokens:
            noun_verb_token_ids = \
                noun_verb_token_ids[:self.const.max_noun_verb_tokens]
        else:
            padding = [-1]*(self.const.max_noun_verb_tokens - num_tokens)
            noun_verb_token_ids = noun_verb_token_ids + padding

        return noun_verb_token_ids

    def get_neg_samples_feats(self,image_id,cap_id,verb_id=None):
        str_image_id = str(image_id)
        str_cap_id = str(cap_id)
        if str_cap_id in self.neg_samples[str_image_id]:
            negs = self.neg_samples[str_image_id][str_cap_id]['negs']
        else:
            feats = np.zeros(
                [self.const.num_neg_verbs,self.const.neg_verb_feat_dim],
                dtype=np.float32)
            verb_id = -1
            return feats, verb_id
        
        if verb_id is None:
            str_verb_id = random.choice(list(negs.keys()))
            verb_id = int(str_verb_id)
        
        feat_name = f'{str_image_id}_{str_cap_id}_{str_verb_id}'
        feats = self.neg_samples_feats[feat_name][()].astype(np.float32)
        return feats, verb_id

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
        
        if self.const.read_noun_verb_tokens is True:
            noun_verb_token_ids = self.noun_verb_token_ids[i]['token_ids']
            to_return['noun_verb_token_ids'] = np.array(
                self.pad_noun_verb_token_ids(noun_verb_token_ids),
                dtype=np.int32)
        
        if self.const.read_neg_samples is True:
            neg_verb_feats, verb_id = self.get_neg_samples_feats(image_id,cap_id)
            to_return['neg_verb_feats'] = neg_verb_feats
            to_return['verb_id'] = np.array(verb_id,dtype=np.int32)

        return to_return

    def get_collate_fn(self):
        def collate_fn(batch):
            new_batch = {}
            for k in batch[0].keys():
                batch_k = [sample[k] for sample in batch]
                if k=='noun_verb_token_ids':
                    new_batch[k] = batch_k
                else:
                    new_batch[k] = default_collate(batch_k)
            
            return new_batch
        
        return collate_fn

if __name__=='__main__':
    const = DetFeatDatasetConstants('val')
    const.read_noun_verb_tokens = True
    const.read_neg_samples = True
    dataset = DetFeatDataset(const)
    print(len(dataset))
    dataloader = DataLoader(
        dataset,
        3,
        num_workers=0,
        collate_fn=dataset.get_collate_fn())
    for data in dataloader:
        import pdb; pdb.set_trace()
