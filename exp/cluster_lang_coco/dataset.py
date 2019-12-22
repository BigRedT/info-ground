import os
import numpy as np
import random
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import utils.io as io
from utils.constants import Constants
from global_constants import coco_paths


class ClusterDatasetConstants(Constants):
    def __init__(self,subset,pos='noun'):
        super().__init__()
        self.subset = subset
        self.pos = pos
        self.feat_h5py = os.path.join(
            coco_paths['local_proc_dir'],
            coco_paths['extracted'][f'{pos}_feats']['feats'][subset])
        self.feat_info_json = os.path.join(
            coco_paths['proc_dir'],
            coco_paths['extracted'][f'{pos}_feats']['feat_info'][subset])
        self.filter_words = None #{'car'}


class ClusterDataset(Dataset):
    def __init__(self,const):
        super().__init__()
        self.const = deepcopy(const)
        self.feat_info = io.load_json_object(self.const.feat_info_json)
        self.filtered_info = self.filter_info()

    def filter_info(self):
        if self.const.filter_words is None:
            filtered_info = []
            for i, info in enumerate(self.feat_info):
                if info is None:
                    continue

                filtered_info.append((i,info))
            
            return filtered_info

        filtered_info = []
        for i, info in enumerate(self.feat_info):
            if info is None:
                continue

            if info['word'] in self.const.filter_words:
                filtered_info.append((i,info))

        return filtered_info

    def __len__(self):
        return len(self.filtered_info)

    def get_feature(self,feat_idx):
        f = io.load_h5py_object(self.const.feat_h5py)
        feat =  f['features'][feat_idx][()].astype(np.float32)
        f.close()
        return feat

    def __getitem__(self, i):
        feat_idx, info = self.filtered_info[i]
        feat = self.get_feature(feat_idx)
        to_return = {
            'feat': feat,
            'cap_id': info['cap_id'],
            'image_id': info['image_id'],
            'caption': info['caption'],
            'word': info['word'],
            'feat_idx': feat_idx,
            'filtered_info_idx': i,
        }
        return to_return


if __name__=='__main__':
    const = ClusterDatasetConstants(subset='train')
    dataset = ClusterDataset(const)
    dataloader = DataLoader(dataset,batch_size=10,shuffle=True,num_workers=10)
    for data in dataloader:
        import pdb; pdb.set_trace()