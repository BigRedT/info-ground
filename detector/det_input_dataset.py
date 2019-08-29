from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import utils.io as io


class DetInputDataset(Dataset):
    def __init__(self,det_input_json):
        super().__init__()
        self.det_input = io.load_json_object(det_input_json)

    def __len__(self):
        return len(self.det_input)

    def read_image(self,img_path):
        image = Image.open(img_path)
        return TF.to_tensor(image)

    def __getitem__(self,i):
        sample = self.det_input[i]
        img_id = sample['id']
        img_path = sample['path']
        img = self.read_image(img_path)
        to_return = {
            'img_id': img_id,
            'img_path': img_path,
            'img': img,
        }
        return to_return

    def create_collate_fn(self):
        def collate_fn(batch):
            if len(batch)==0:
                return None
            
            collated_batch = {}
            for k in batch[0].keys():
                if k=='img':
                    collator = lambda x: x
                else:
                    collator = lambda x: default_collate(x)

                collated_batch[k] = collator([sample[k] for sample in batch])

            return collated_batch

        return collate_fn
                


if __name__=='__main__':
    det_input_json = '/home/tgupta6/Data/coco_proc/det_input_train.json'
    dataset = DetInputDataset(det_input_json)
    import pdb; pdb.set_trace()