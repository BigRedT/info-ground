import os
import numpy as np
import random
from PIL import Image
from copy import deepcopy
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import utils.io as io
from utils.constants import Constants
from global_constants import coco_paths


class SelfSupDetFeatDatasetConstants(Constants):
    def __init__(self,subset):
        super().__init__()
        self.subset = subset
        self.subset_image_dirname = coco_paths['extracted']['images'][subset]

        self.image_dir = coco_paths['image_dir']

        # Detected regions, features, labels, scores
        self.det_dir = os.path.join(
            coco_paths['local_proc_dir'],
            f'detections/{subset}')
        self.boxes_hdf5 = os.path.join(self.det_dir,'boxes.hdf5')
        self.features_hdf5 = os.path.join(self.det_dir,'features.hdf5')
        self.labels_hdf5 = os.path.join(self.det_dir,'labels.hdf5')
        self.scores_hdf5 = os.path.join(self.det_dir,'scores.hdf5')

        self.self_sup_features_hdf5 = os.path.join(
            coco_paths['local_proc_dir'],
            coco_paths['extracted']['self_sup_feats'][subset])
        
        # Caption annos
        self.annos_json = os.path.join(
            coco_paths['proc_dir'],
            coco_paths['extracted']['annos']['captions'][subset])
        self.max_objects = 15
        self.mask_prob = 0.2

        # Noun and verb tokens in captions for MI training
        self.noun_verb_tokens_json = os.path.join(
            coco_paths['proc_dir'],
            coco_paths['extracted']['annos']['noun_verb_tokens'][subset])
        self.read_noun_verb_tokens = True
        self.max_noun_verb_tokens = 6
        
        # Negative noun samples
        self.neg_noun_samples_json = os.path.join(
            coco_paths['proc_dir'],
            coco_paths['extracted']['noun_negatives']['samples'][subset])
        self.neg_noun_samples_h5py = os.path.join(
            coco_paths['local_proc_dir'],
            coco_paths['extracted']['noun_negatives']['feats'][subset])
        self.read_neg_noun_samples = True
        self.num_neg_nouns = 25
        self.neg_noun_feat_dim = 768

        self.image_size = [224,224]


class SelfSupDetFeatDataset(Dataset):
    def __init__(self,const):
        self.const = deepcopy(const)
        self.annos = io.load_json_object(self.const.annos_json)
        
        if self.const.read_noun_verb_tokens is True:
            self.noun_verb_token_ids = io.load_json_object(
                self.const.noun_verb_tokens_json)
        
        if self.const.read_neg_noun_samples is True:
            self.neg_noun_samples = io.load_json_object(
                self.const.neg_noun_samples_json)

        normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        self.transforms = transforms.Compose([
            transforms.Resize(self.const.image_size),
            transforms.ToTensor(),
            normalize])
        os.environ['HDF5_USE_FILE_LOCKING']="FALSE"
        
    def get_image_name(self,subset,image_id):
        image_id = str(image_id).zfill(12)
        return f'COCO_{self.const.subset_image_dirname}_{image_id}'
    
    def get_image_path(self,image_id):
        image_name = self.get_image_name(self.const.subset,image_id)
        return os.path.join(
            os.path.join(self.const.image_dir,self.const.subset_image_dirname),
            f'{image_name}.jpg')
    
    def read_image(self,img_path):
        image = Image.open(img_path).convert(mode='RGB') # PIL image
        return image

    def __len__(self):
        return len(self.annos['annotations'])

    def read_object_features(self,image_name):
        f = io.load_h5py_object(self.const.features_hdf5)
        features = f[image_name][()]
        f.close()
        return features

    def read_self_sup_features(self,image_name):
        f = io.load_h5py_object(self.const.self_sup_features_hdf5)
        if image_name not in f:
            return None

        features = f[image_name][()]
        f.close()
        return features
    
    def read_boxes(self,image_name):
        f = io.load_h5py_object(self.const.boxes_hdf5)
        boxes = f[image_name][()]
        f.close()
        return boxes
    
    def scale_boxes(self,boxes,H,W,h,w):
        """
        H,W: original image size
        h,w: image size to which to scale the boxes
        """
        for i in [0,2]:
            boxes[:,i] = boxes[:,i]*w/W
            boxes[:,i+1] = boxes[:,i+1]*h/H

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

    def get_neg_noun_samples_feats(self,image_id,cap_id,noun_id=None):
        str_image_id = str(image_id)
        str_cap_id = str(cap_id)
        
        if (str_image_id in self.neg_noun_samples) and \
            (str_cap_id in self.neg_noun_samples[str_image_id]):
            negs = self.neg_noun_samples[str_image_id][str_cap_id]['negs']
        else:
            feats = np.zeros(
                [1+self.const.num_neg_nouns,self.const.neg_noun_feat_dim],
                dtype=np.float32)
            noun_id = -1
            return feats, noun_id
        
        if noun_id is None:
            str_noun_id = random.choice(list(negs.keys()))
            noun_id = int(str_noun_id)
        
        neg_samples_feats = io.load_h5py_object(self.const.neg_noun_samples_h5py)
        feat_name = f'{str_image_id}_{str_cap_id}_{str_noun_id}'
        feats = neg_samples_feats[feat_name][()].astype(np.float32)
        neg_samples_feats.close()

        return feats, noun_id

    def __getitem__(self, i):
        anno = self.annos['annotations'][i]
        image_id = anno['image_id']
        cap_id = anno['id']
        caption = anno['caption']
        image_name = self.get_image_name(self.const.subset,anno['image_id'])
        image_path = self.get_image_path(image_id)
        image = self.read_image(image_path)
        W,H = image.size
        image = self.transforms(image)
        boxes = self.read_boxes(image_name)
        boxes = self.scale_boxes(boxes,H,W,28,28)
        features = self.read_object_features(image_name)
        self_sup_features = self.read_self_sup_features(image_name)
        if self_sup_features is None:
            self_sup_features = np.zeros([features.shape[0],256])
            
        num_objects = features.shape[0]
        features, pad_mask = self.pad_object_features(features)
        self_sup_features, _ = self.pad_object_features(self_sup_features)
        features = np.concatenate((features,self_sup_features),1)
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
            'image': image,
            'boxes': boxes,
        }
        
        if self.const.read_noun_verb_tokens is True:
            noun_verb_token_ids = self.noun_verb_token_ids[i]['token_ids']
            to_return['noun_verb_token_ids'] = np.array(
                self.pad_noun_verb_token_ids(noun_verb_token_ids),
                dtype=np.int32)

        if self.const.read_neg_noun_samples is True:
            neg_noun_feats, noun_id = self.get_neg_noun_samples_feats(image_id,cap_id)
            to_return['neg_noun_feats'] = neg_noun_feats
            to_return['noun_id'] = np.array(noun_id,dtype=np.int32)

        return to_return

    def get_collate_fn(self):
        def collate_fn(batch):
            new_batch = {}
            for k in batch[0].keys():
                batch_k = [sample[k] for sample in batch]
                if k=='boxes':
                    new_batch[k] = [torch.FloatTensor(s) for s in batch_k]
                else:
                    try:
                        new_batch[k] = default_collate(batch_k)
                    except:
                        import pdb; pdb.set_trace()
            
            return new_batch
        
        return collate_fn

if __name__=='__main__':
    const = SelfSupDetFeatDatasetConstants('val')
    dataset = SelfSupDetFeatDataset(const)
    print(len(dataset))
    dataloader = DataLoader(
        dataset,
        50,
        num_workers=0,
        collate_fn=dataset.get_collate_fn())
    for data in dataloader:
        import pdb; pdb.set_trace()
