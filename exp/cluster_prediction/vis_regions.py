import os
import h5py
import math
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import itertools
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import skimage.io as skio

from utils.bbox_utils import vis_bbox
import utils.io as io
from utils.constants import save_constants, Constants
from .dataset import ClusterPredDataset
from detector.model import create_detector


def select_boxes(logits,boxes,gt_labels,label_to_idx):
    selected_ids = torch.topk(logits,3,0)[1]
    label_to_boxes = {}
    for gt_label in gt_labels:
        box_ids = selected_ids[:,label_to_idx[gt_label]]
        label_to_boxes[gt_label] = boxes[box_ids].detach().cpu().numpy()

    return label_to_boxes


def visualize(model,dataloader,exp_const):
    for it,data in enumerate(dataloader):
        if data is None:
            continue
        
        # Set mode
        model.detector.eval()

        images = [img.cuda() for img in data['image']]
        label_vec = [vec.cuda() for vec in data['label_vec']]
        boxes = [b.cuda() for b in data['boxes']]

        logits,features = model.detector(images,boxes)
        label_to_boxes = select_boxes(
            logits[0],
            boxes[0],
            data['labels'][0],
            dataloader.dataset.label_to_idx)

        image_id = data['image_id'][0]
        vis_dir = os.path.join(
            exp_const.vis_dir,
            f'regions/{image_id}')
        io.mkdir_if_not_exists(vis_dir,recursive=True)

        image_path = data['image_path'][0]
        image = skio.imread(image_path)
        for label, selected_boxes in label_to_boxes.items():
            box_image = copy.deepcopy(image)
            for k in range(selected_boxes.shape[0]):
                box = selected_boxes[k]
                box_image = vis_bbox(box,box_image,modify=True)

            filename = os.path.join(vis_dir,f'{label}.jpg')
            skio.imsave(filename,box_image)
            
        import pdb; pdb.set_trace()
            

def main(exp_const,data_const,model_const):
    np.random.seed(exp_const.seed)
    torch.manual_seed(exp_const.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print('Creating dataloader ...')
    dataset = ClusterPredDataset(data_const)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        collate_fn=dataset.create_collate_fn())

    print('Creating network ...')
    model = Constants()
    model.const = model_const
    model.num_classes = len(dataloader.dataset.labels)
    model.detector = create_detector(
        extractor=True,
        num_classes=model.num_classes)
    model.detector.load_state_dict(
        torch.load(model.const.detector_path)['state_dict'])
    model.detector.cuda()

    visualize(model,dataloader,exp_const)