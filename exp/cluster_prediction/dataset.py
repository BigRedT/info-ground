from PIL import Image
import os
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import utils.io as io
from global_constants import coco_paths


class ClusterPredDatasetConstants(io.JsonSerializableClass):
    def __init__(self,subset):
        super().__init__()
        self.subset = subset
        self.cluster_dir = coco_paths['clusters']['dir']
        
        self.image_dir = os.path.join(
            coco_paths['image_dir'],
            coco_paths['extracted']['images'][self.subset])
        image_subdir = coco_paths['extracted']['images'][self.subset]
        self.image_name_prefix = f'COCO_{image_subdir}'

        self.image_labels_json = os.path.join(
            self.cluster_dir,
            coco_paths['clusters']['image_labels'][self.subset])
        # labels_json is the list of all labels and is always set to train
        self.labels_json = os.path.join(
            self.cluster_dir,
            coco_paths['clusters']['labels']['train'])
        
        self.det_dir = os.path.join(
            coco_paths['proc_dir'],
            f'detections/{subset}')
        self.boxes_hdf5 = os.path.join(self.det_dir,'boxes.hdf5')
        self.scores_hdf5 = os.path.join(self.det_dir,'scores.hdf5')
        self.labels_hdf5 = os.path.join(self.det_dir,'labels.hdf5')


class ClusterPredDataset(Dataset):
    def __init__(self,const):
        super().__init__()
        self.const = const
        self.image_labels = io.load_json_object(self.const.image_labels_json)
        self.image_labels = list(self.image_labels.items())
        self.labels = io.load_json_object(self.const.labels_json)
        self.label_to_idx = {l:i for i,l in enumerate(self.labels)}

    def __len__(self):
        return len(self.image_labels)

    def read_image(self,image_id):
        img_path = self.image_id_to_path(image_id)
        image = Image.open(img_path)
        return TF.to_tensor(image)
    
    def image_id_to_path(self,image_id):
        image_id_str = str(image_id).zfill(12)
        image_path = os.path.join(
            self.const.image_dir,
            f'{self.const.image_name_prefix}_{image_id_str}.jpg')
        return image_path

    def get_label_vector(self,labels):
        #labels = [l for l in labels if '-1' not in l]
        word_to_idx = {}
        for label in labels:
            word = label.split('_')[0]
            if word not in word_to_idx:
                word_to_idx[word] = []
            word_to_idx[word].append(self.label_to_idx[label])

        label_vec = np.zeros(len(self.labels),dtype=np.float32)
        for word,ids in word_to_idx.items():
            value = 1/len(ids)
            for idx in ids:
                label_vec[idx] = value
        
        # label_vec = np.zeros(len(self.labels),dtype=np.float32)
        # for label in labels:
        #     idx = self.label_to_idx[label]
        #     label_vec[idx] = 1
        
        # label_vec = label_vec / np.sum(label_vec)
        return label_vec

    def read_boxes(self,image_id):
        image_id_str = str(image_id).zfill(12)
        image_name = f'{self.const.image_name_prefix}_{image_id_str}'
        f = io.load_h5py_object(self.const.boxes_hdf5)
        boxes = f[image_name][()]
        f.close()
        return boxes
    
    def read_scores(self,image_id):
        image_id_str = str(image_id).zfill(12)
        image_name = f'{self.const.image_name_prefix}_{image_id_str}'
        f = io.load_h5py_object(self.const.scores_hdf5)
        scores = f[image_name][()]
        f.close()
        return scores

    def read_det_labels(self,image_id):
        image_id_str = str(image_id).zfill(12)
        image_name = f'{self.const.image_name_prefix}_{image_id_str}'
        f = io.load_h5py_object(self.const.labels_hdf5)
        labels = f[image_name][()]
        f.close()
        return labels

    def get_det_label_vector(self,det_labels,scores):
        B = scores.shape[0]
        L = 91
        vec = np.zeros([B,L],dtype=np.float32)
        for i in range(B):
            vec[i,det_labels[i]] = scores[i]
        
        return vec
        
    def __getitem__(self,i):
        image_id, labels = self.image_labels[i]
        image = self.read_image(image_id)
        boxes = self.read_boxes(image_id)
        scores = self.read_scores(image_id)
        det_labels = self.read_det_labels(image_id)
        if boxes.shape[0]==0:
            return None

        if boxes.shape[0] > 20:
            boxes = boxes[:20]
            scores = scores[:20]
            det_labels = det_labels[:20]

        label_vec = self.get_label_vector(labels)
        det_label_vec = self.get_det_label_vector(det_labels,scores)

        image_path = self.image_id_to_path(image_id)

        to_return = {
            'image_path': image_path,
            'image_id': image_id,
            'image': image,
            'labels': labels,
            'label_vec': label_vec,
            'boxes': boxes,
            'scores': scores,
            'det_labels': det_labels,
            'det_label_vec': det_label_vec,
        }
        return to_return


    def create_collate_fn(self):
        def collate_fn(batch):
            batch = [s for s in batch if s is not None]
            
            if len(batch)==0:
                return None
            
            collated_batch = {}
            for k in batch[0].keys():
                collated_batch[k] = [sample[k] for sample in batch]

            for k in ['image','label_vec','boxes','scores','det_label_vec']:
                collated_batch[k] = [
                    torch.FloatTensor(s) for s in collated_batch[k]]
            return collated_batch

        return collate_fn
                


if __name__=='__main__':
    const = ClusterPredDatasetConstants('train')
    dataset = ClusterPredDataset(const)
    dataloader = DataLoader(
        dataset,
        batch_size=5,
        collate_fn=dataset.create_collate_fn())
    for data in dataloader:
        import pdb; pdb.set_trace()