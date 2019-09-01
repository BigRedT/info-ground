import os
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
            coco_paths['proc_dir'],
            f'detections/{subset}')
        self.boxes_hdf5 = os.path.join(self.det_dir,'boxes.hdf5')
        self.features_hdf5 = os.path.join(self.det_dir,'features.hdf5')
        self.labels_hdf5 = os.path.join(self.det_dir,'labels.hdf5')
        self.scores_hdf5 = os.path.join(self.det_dir,'scores.hdf5')
        self.annos_json = os.path.join(
            coco_paths['proc_dir'],
            coco_paths['extracted']['annos']['captions'][subset]) 


class DetFeatDataset(Dataset):
    def __init__(self,const):
        self.const = deepcopy(const)
        self.annos = io.load_json_object(self.const.annos_json)
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

    def __getitem__(self, i):
        anno = self.annos['annotations'][i]
        image_id = anno['image_id']
        cap_id = anno['id']
        caption = anno['caption']
        image_name = self.get_image_name(self.const.subset,anno['image_id'])
        features = self.read_object_features(image_name)
        to_return = {
            'image_id': image_id,
            'cap_id': cap_id,
            'image_name': image_name,
            'caption': caption,
            'features': features
        }
        return to_return


if __name__=='__main__':
    const = DetFeatDatasetConstants('val')
    dataset = DetFeatDataset(const)
    dataset[0]
    import pdb; pdb.set_trace()
