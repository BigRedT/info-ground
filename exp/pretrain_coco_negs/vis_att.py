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

import utils.io as io
from utils.constants import save_constants, Constants
from .models.object_encoder import ObjectEncoder
from .models.cap_encoder import CapEncoder
from .dataset import DetFeatDataset
from .models.factored_cap_info_nce_loss import CapInfoNCE, KLayer, FLayer
from utils.bbox_utils import vis_bbox, create_att
from utils.html_writer import HtmlWriter


def create_info_nce_criterion(x_dim,c_dim,d):
    fx = nn.Sequential(
        nn.Linear(x_dim,d))

    fy = nn.Sequential(
        nn.Linear(c_dim,d))

    criterion = InfoNCE(fx,fy)
    
    return criterion


def create_cap_info_nce_criterion(o_dim,u_dim,w_dim,d):
    fo = FLayer(o_dim,d)
    fw = FLayer(w_dim,d)
    ku = KLayer(u_dim,d)
    kw = KLayer(w_dim,d)
    criterion = CapInfoNCE(fo,fw,ku,kw)
    
    return criterion


def eval_model(model,dataloader,exp_const):
    # Set mode
    model.object_encoder.eval()
    model.cap_encoder.eval()
    model.lang_sup_criterion.eval()

    num_samples = 0
    for it,data in enumerate(tqdm(dataloader)):
        if (exp_const.num_vis_samples is not None) and \
            (num_samples >= exp_const.num_vis_samples):
                break

        # Forward pass
        object_features = data['features'].cuda()
        object_mask = data['object_mask'].cuda()
        pad_mask = data['pad_mask'].cuda()
        context_object_features = model.object_encoder(
            object_features)
        
        token_ids, tokens, token_lens = model.cap_encoder.tokenize_batch(
            data['caption'])
        token_ids = torch.LongTensor(token_ids).cuda()
        token_features = model.cap_encoder(token_ids)

        token_mask = model.cap_encoder.get_token_mask(tokens)
        token_mask = torch.cuda.FloatTensor(token_mask)
        lang_sup_loss, att, att_V_o = model.lang_sup_criterion(
            context_object_features,
            object_features,
            token_features,
            token_mask)

        num_objects = data['num_objects'][0].item()
        num_tokens = len(tokens[0])
        att = att[0,0]
        att = att[:num_tokens,:num_objects]
        
        if num_objects >= 3:
            K = 3
        else:
            K = num_objects
        
        _, indices = torch.topk(att,K,1)
        box_ids = indices.detach().cpu().numpy()
        att = att.detach().cpu().numpy()
        
        image_name = data['image_name'][0]
        boxes = dataloader.dataset.read_boxes(image_name)

        vis_dir = os.path.join(exp_const.vis_dir,image_name)
        io.mkdir_if_not_exists(vis_dir,recursive=True)

        html_writer = HtmlWriter(os.path.join(vis_dir,'index.html'))

        src_filename = os.path.join(
            dataloader.dataset.const.image_dir,
            image_name + '.jpg')
        img = skio.imread(src_filename)

        box_img = img
        num_boxes = min(15,len(boxes))
        for b in range(num_boxes):
            box_img = vis_bbox(
                boxes[b],
                box_img,
                (255,0,0),
                modify=False)
        
        dst_filename = os.path.join(vis_dir,'all_boxes.jpg')
        skio.imsave(dst_filename,box_img)
        col_dict = {
            0: 'All Boxes',
            1: html_writer.image_tag('all_boxes.jpg'),
            2: f'Total number of boxes: {num_boxes}'
        }
        html_writer.add_element(col_dict)

        colors = [
            (255,0,0),
            (0,255,0),
            (0,0,255)]
        for i in range(num_tokens):
            box_img = 0*img
            box_img = box_img.astype(np.float32)
            for k in range(K):
                if box_ids[i,k] >= len(boxes):
                    break
                
                box_img = create_att(
                    boxes[box_ids[i,k]],
                    box_img,
                    np.minimum(1,att[i,box_ids[i,k]]**0.5))
            
            box_img = box_img*img.astype(np.float32)
            box_img = box_img.astype(np.uint8)

            for k in range(K):
                box_img = vis_bbox(
                    boxes[box_ids[i,k]],
                    box_img,
                    colors[k],
                    modify=False,
                    alpha=0)

            dst_filename = os.path.join(vis_dir,str(i)+'.jpg')
            skio.imsave(dst_filename,box_img)
            col_dict = {
                0: tokens[0][i],
                1: html_writer.image_tag(str(i)+'.jpg'),
                2: att[i,box_ids[i]],
            }
            html_writer.add_element(col_dict)

        html_writer.close()
        num_samples += 1


def main(exp_const,data_const,model_const):
    np.random.seed(exp_const.seed)
    torch.manual_seed(exp_const.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)
    io.mkdir_if_not_exists(exp_const.log_dir)
    io.mkdir_if_not_exists(exp_const.model_dir)
    io.mkdir_if_not_exists(exp_const.vis_dir)
    
    print('Creating network ...')
    model = Constants()
    model.const = model_const
    model.object_encoder = ObjectEncoder(model.const.object_encoder)
    model.cap_encoder = CapEncoder(model.const.cap_encoder)
    model.lang_sup_criterion = create_cap_info_nce_criterion(
        model.object_encoder.const.context_layer.hidden_size,
        model.object_encoder.const.object_feature_dim,
        model.cap_encoder.model.config.hidden_size,
        model.cap_encoder.model.config.hidden_size//2)
    if model.const.model_num != -1:
        print('Loading model num',model.const.model_num,'...')
        loaded_object_encoder = torch.load(model.const.object_encoder_path)
        print(loaded_object_encoder['step'])
        model.object_encoder.load_state_dict(
            loaded_object_encoder['state_dict'])
        model.lang_sup_criterion.load_state_dict(
            torch.load(model.const.lang_sup_criterion_path)['state_dict'])
    model.object_encoder.cuda()
    model.cap_encoder.cuda()
    model.lang_sup_criterion.cuda()

    print('Creating dataloader ...')
    dataloaders = {}
    dataset = DetFeatDataset(data_const)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1)

    eval_model(model,dataloader,exp_const)