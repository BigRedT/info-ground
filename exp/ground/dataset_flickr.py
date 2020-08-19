import os
import numpy as np
import random
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import utils.io as io
from utils.constants import Constants
from global_constants import flickr_paths


class FlickrDatasetConstants(Constants):
    def __init__(self,subset):
        super().__init__()
        self.subset = subset
        self.det_dir = os.path.join(flickr_paths['det_dir'],self.subset)
        self.image_dir = flickr_paths['image_dir']

        self.image_ids_txt = os.path.join(
            flickr_paths['downloads_dir'],
            flickr_paths['subsets'][self.subset])
        self.phrase_boxes_json = os.path.join(
            flickr_paths['proc_dir'],
            flickr_paths['phrase_boxes'][self.subset])
        self.sentences_json = os.path.join(
            flickr_paths['proc_dir'],
            flickr_paths['sentences'][self.subset])
        
        self.boxes_hdf5 = os.path.join(self.det_dir,'boxes.hdf5')
        self.features_hdf5 = os.path.join(self.det_dir,'features.hdf5')
        self.labels_hdf5 = os.path.join(self.det_dir,'labels.hdf5')
        self.scores_hdf5 = os.path.join(self.det_dir,'scores.hdf5')

        self.max_objects = 30
        self.mask_prob = 0.2

        # Noun and adj tokens in captions for MI training
        self.noun_adj_tokens_json = os.path.join(
            flickr_paths['proc_dir'],
            flickr_paths['noun_adj_tokens'][subset])
        self.read_noun_adj_tokens = True
        self.max_noun_adj_tokens = 6
        
        # Negative noun samples
        self.neg_noun_samples_json = os.path.join(
            flickr_paths['proc_dir'],
            flickr_paths['noun_negatives']['samples'][subset])
        self.neg_noun_samples_h5py = os.path.join(
            flickr_paths['local_proc_dir'],
            flickr_paths['noun_negatives']['feats'][subset])
        self.read_neg_noun_samples = True
        self.num_neg_nouns = 25
        self.neg_noun_feat_dim = 768


class FlickrDataset(Dataset):
    def __init__(self,const):
        super().__init__()
        self.const = deepcopy(const)
        self.image_ids = self.read_image_ids()
        self.phrase_boxes = io.load_json_object(self.const.phrase_boxes_json)
        self.sentences = io.load_json_object(self.const.sentences_json)
        if self.const.read_noun_adj_tokens is True:
            self.noun_adj_token_ids = io.load_json_object(
                self.const.noun_adj_tokens_json)
        
        if self.const.read_neg_noun_samples is True:
            self.neg_noun_samples = io.load_json_object(
                self.const.neg_noun_samples_json)
        os.environ['HDF5_USE_FILE_LOCKING']="FALSE"

    def read_image_ids(self):
        image_ids = io.read(self.const.image_ids_txt)
        image_ids = [idx.decode() for idx in image_ids.split()]
        return image_ids

    def get_image_path(self,image_id):
        return os.path.join(
            self.const.image_dir,
            f'{image_id}.jpg')
            
    def __len__(self):
        return 5*len(self.image_ids)

    def get_caption(self,image_id,cap_num):
        return self.sentences[image_id][cap_num]['sentence']

    def get_phrases(self,image_id,cap_num):
        return self.sentences[image_id][cap_num]['phrases']

    def get_gt_boxes(self,image_id):
        return self.phrase_boxes[image_id]

    def read_object_features(self,image_id):
        f = io.load_h5py_object(self.const.features_hdf5)
        features = f[image_id][()]
        f.close()
        return features

    def read_object_boxes(self,image_id):
        f = io.load_h5py_object(self.const.boxes_hdf5)
        boxes = f[image_id][()]
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

    def pad_noun_adj_token_ids(self,noun_adj_token_ids):
        num_tokens = len(noun_adj_token_ids)
        if num_tokens >= self.const.max_noun_adj_tokens:
            noun_adj_token_ids = \
                noun_adj_token_ids[:self.const.max_noun_adj_tokens]
        else:
            padding = [-1]*(self.const.max_noun_adj_tokens - num_tokens)
            noun_adj_token_ids = noun_adj_token_ids + padding

        return noun_adj_token_ids

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

    def __getitem__(self,i):
        image_num = i//5
        cap_num = i%5
        image_id = self.image_ids[image_num]
        cap_id = f'{image_id}_{cap_num}'
        caption = self.get_caption(image_id,cap_num)
        #phrases = self.get_phrases(image_id,cap_num)
        features = self.read_object_features(image_id)
        num_objects = features.shape[0]
        features, pad_mask = self.pad_object_features(features)
        object_mask = self.mask_objects(num_objects)
        #boxes = self.read_object_boxes(image_id)
        #gt_boxes = self.get_gt_boxes(image_id)
        to_return = {
            'image_id': image_id,
            'cap_num': cap_num,
            'cap_id': cap_id,
            'caption': caption,
            #'phrases': phrases,
            'features': features,
            'object_mask': object_mask,
            #'boxes': boxes,
            #'gt_boxes': gt_boxes,
            'pad_mask': pad_mask,
        }

        if self.const.read_noun_adj_tokens is True:
            noun_adj_token_ids = self.noun_adj_token_ids[i]['token_ids']
            to_return['noun_adj_token_ids'] = np.array(
                self.pad_noun_adj_token_ids(noun_adj_token_ids),
                dtype=np.int32)

        if self.const.read_neg_noun_samples is True:
            neg_noun_feats, noun_id = self.get_neg_noun_samples_feats(image_id,cap_id)
            to_return['neg_noun_feats'] = neg_noun_feats
            to_return['noun_id'] = np.array(noun_id,dtype=np.int32)

        return to_return


if __name__=='__main__':
    const = FlickrDatasetConstants('val')
    dataset = FlickrDataset(const)
    print(len(dataset),len(dataset)//5,5*(len(dataset)//5))
    import pdb; pdb.set_trace()