import os
import numpy as np
import random
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import utils.io as io
from utils.constants import Constants
from global_constants import hico_paths


class HICOFeatDatasetConstants(Constants):
    def __init__(self,subset):
        super().__init__()
        self.subset = subset
        self.hoi_list_json = os.path.join(
            hico_paths['proc_dir'],
            hico_paths['hoi_list_json'])
        self.subset_list_json = os.path.join(
            hico_paths['proc_dir'],
            hico_paths['subset_list_json'][subset])
        self.labels_npy = os.path.join(
            hico_paths['proc_dir'],
            hico_paths['labels_npy'][subset])
        
        # Detections
        self.det_dir = os.path.join(hico_paths['detection_dir'],subset)
        self.boxes_hdf5 = os.path.join(self.det_dir,'boxes.hdf5')
        self.features_hdf5 = os.path.join(self.det_dir,'features.hdf5')
        self.labels_hdf5 = os.path.join(self.det_dir,'labels.hdf5')
        self.scores_hdf5 = os.path.join(self.det_dir,'scores.hdf5')

        self.max_objects = 15
        self.mask_prob = 0.2


class HICOFeatDataset(Dataset):
    def __init__(self,const):
        super().__init__()
        self.const = deepcopy(const)
        self.hoi_list = io.load_json_object(self.const.hoi_list_json)
        self.subset_list = io.load_json_object(self.const.subset_list_json)
        self.labels = np.load(self.const.labels_npy)
        
        os.environ['HDF5_USE_FILE_LOCKING']="FALSE"

    def __len__(self):
        return len(self.subset_list)

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

    def get_pos_neg_unk_labels(self,labels):
        pos_labels = (labels==1).astype(np.float32)
        neg_labels = (labels==-1).astype(np.float32)
        unk_labels = np.isnan(labels).astype(np.float32)
        return pos_labels, neg_labels, unk_labels

    def __getitem__(self, i):
        image_name = self.subset_list[i]
        labels = self.labels[i]
        pos_labels, neg_labels, unk_labels = self.get_pos_neg_unk_labels(labels)
        features = self.read_object_features(image_name)
        num_objects = features.shape[0]
        features, pad_mask = self.pad_object_features(features)
        object_mask = self.mask_objects(num_objects)
        to_return = {
            'idx': i,
            'image_name': image_name,
            'features': features,
            'num_objects': num_objects,
            'object_mask': object_mask,
            'pad_mask': pad_mask,
            'pos_labels': pos_labels,
            'neg_labels': neg_labels,
            'unk_labels': unk_labels,
        }
        return to_return


if __name__=='__main__':
    const = HICOFeatDatasetConstants('val')
    dataset = HICOFeatDataset(const)
    print(len(dataset))
    dataloader = DataLoader(
        dataset,
        50,
        num_workers=0)
    for data in dataloader:
        import pdb; pdb.set_trace()
        


