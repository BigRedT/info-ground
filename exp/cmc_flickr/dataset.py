import os
#from multiprocessing import Pool
from PIL import Image
from skimage import color
import numpy as np
import torch
from copy import deepcopy
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import utils.io as io
from global_constants import flickr_paths


class RGB2Lab(object):
    """Convert RGB PIL image to ndarray Lab."""
    def __call__(self, img):
        img = np.array(img) # uint8 image
        img = color.rgb2lab(img)
        return img


class FlickrDatasetConstants(io.JsonSerializableClass):
    def __init__(self,subset):
        super().__init__()
        self.subset = subset
        self.image_dir = flickr_paths['image_dir']

        self.det_dir = os.path.join(flickr_paths['det_dir'],self.subset)
        self.boxes_hdf5 = os.path.join(self.det_dir,'boxes.hdf5')

        self.image_size = (224,224)


def crop_box(image,box,size,transforms):
    crop = image.crop(box).resize(size)
    return transforms(crop)


class FlickrDataset(Dataset):
    def __init__(self,const):
        self.const = deepcopy(const)
        self.image_ids = self.get_image_ids()
        self.mean = [(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2]
        self.std = [(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]
        self.transforms = transforms.Compose([
            RGB2Lab(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        #self.p = Pool(10)


    def get_image_ids(self):
        image_ids = list(io.load_h5py_object(self.const.boxes_hdf5).keys())
        return sorted(image_ids)

    def get_image_path(self,image_id):
        return os.path.join(
            self.const.image_dir,
            f'{image_id}.jpg')
    
    def read_image(self,img_path):
        image = Image.open(img_path).convert(mode='RGB') # PIL image
        return image

    def crop(self,image,boxes):
        crops = []
        for box in boxes:
            crop = image.crop(box).resize(self.const.image_size)
            crops.append(self.transforms(crop))

        crops = torch.stack(crops)
        return crops
    
    # def crop_parallel(self,image,boxes):
    #     crop_args = []
    #     for box in boxes:
    #         crop_args.append((image,box,self.const.image_size,self.transforms))
        
    #     crops = self.p.starmap(crop_box,crop_args)
    #     crops = torch.stack(crops)
    #     return crops
    
    def read_boxes(self,image_id):
        f = io.load_h5py_object(self.const.boxes_hdf5)
        boxes = f[image_id][()]
        f.close()
        return boxes

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, i):
        image_id = self.image_ids[i]
        image_path = self.get_image_path(image_id)
        image = self.read_image(image_path)
        boxes = self.read_boxes(image_id)
        if boxes.shape[0]==0:
            return None

        crops = self.crop(image,boxes[:30])
        to_return = {
            'image_id': image_id,
            'num_crops': crops.size(0),
            'crops': crops,
        }
        return to_return

    def get_collate_fn(self):
        def collate_fn(batch):
            batch = [s for s in batch if s is not None]
            if len(batch)==0:
                return None

            new_batch = {}
            for k in batch[0].keys():
                new_batch[k] = [s[k] for s in batch]
            
            new_batch['crops'] = torch.cat(new_batch['crops'],0)
            
            return new_batch

        return collate_fn


if __name__=='__main__':
    const = FlickrDatasetConstants('test')
    dataset = FlickrDataset(const)
    dataloader = DataLoader(
        dataset,
        batch_size=10,
        num_workers=10,
        collate_fn=dataset.get_collate_fn())
    for data in dataloader: 
        import pdb; pdb.set_trace()