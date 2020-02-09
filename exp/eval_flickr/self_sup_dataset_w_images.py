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
from global_constants import flickr_paths


class SelfSupFlickrDatasetConstants(Constants):
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

        self.self_sup_features_hdf5 = os.path.join(
            flickr_paths['local_proc_dir'],
            flickr_paths['self_sup_feats'][subset])

        self.max_objects = 15 # set to 30 for final exps        

        self.image_size = [224,224]


class SelfSupFlickrDataset(Dataset):
    def __init__(self,const):
        super().__init__()
        self.const = deepcopy(const)
        self.image_ids = self.read_image_ids()
        self.phrase_boxes = io.load_json_object(self.const.phrase_boxes_json)
        self.sentences = io.load_json_object(self.const.sentences_json)
        
        normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        self.transforms = transforms.Compose([
            transforms.Resize(self.const.image_size),
            transforms.ToTensor(),
            normalize])
        os.environ['HDF5_USE_FILE_LOCKING']="FALSE"

    def read_image_ids(self):
        image_ids = io.read(self.const.image_ids_txt)
        image_ids = [idx.decode() for idx in image_ids.split()]
        return image_ids

    def get_image_path(self,image_id):
        return os.path.join(
            self.const.image_dir,
            f'{image_id}.jpg')
    
    def read_image(self,img_path):
        image = Image.open(img_path).convert(mode='RGB') # PIL image
        return image

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

    def read_self_sup_features(self,image_id):
        f = io.load_h5py_object(self.const.self_sup_features_hdf5)
        if image_id not in f:
            return None

        features = f[image_id][()]
        f.close()
        return features

    def read_object_boxes(self,image_id):
        f = io.load_h5py_object(self.const.boxes_hdf5)
        boxes = f[image_id][()]
        f.close()
        return boxes

    def scale_boxes(self,boxes,H,W,h,w):
        """
        H,W: original image size
        h,w: image size to which to scale the boxes
        """
        scaled_boxes = np.copy(boxes)
        for i in [0,2]:
            scaled_boxes[:,i] = boxes[:,i]*w/W
            scaled_boxes[:,i+1] = boxes[:,i+1]*h/H

        return scaled_boxes

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

    def __getitem__(self,i):
        image_num = i//5
        cap_num = i%5
        image_id = self.image_ids[image_num]
        image_path = self.get_image_path(image_id)
        image = self.read_image(image_path)
        W,H = image.size
        image = self.transforms(image)
        caption = self.get_caption(image_id,cap_num)
        phrases = self.get_phrases(image_id,cap_num)
        features = self.read_object_features(image_id)
        self_sup_features = self.read_self_sup_features(image_id)
        if self_sup_features is None:
            self_sup_features = np.zeros([features.shape[0],256])

        features, pad_mask = self.pad_object_features(features)
        self_sup_features, _ = self.pad_object_features(self_sup_features)
        features = np.concatenate((features,self_sup_features),1)

        boxes = self.read_object_boxes(image_id)
        scaled_boxes = self.scale_boxes(boxes,H,W,28,28)
        gt_boxes = self.get_gt_boxes(image_id)
        to_return = {
            'image_id': image_id,
            'cap_num': cap_num,
            'caption': caption,
            'phrases': phrases,
            'features': features,
            'boxes': boxes,
            'gt_boxes': gt_boxes,
            'pad_mask': pad_mask,
            'image': image,
            'scaled_boxes': scaled_boxes,
        }
        return to_return


if __name__=='__main__':
    const = SelfSupFlickrDatasetConstants('test')
    dataset = SelfSupFlickrDataset(const)
    import pdb; pdb.set_trace()