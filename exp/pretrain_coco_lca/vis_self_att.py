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
from .models.info_nce_loss import InfoNCE
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


def agg_attention(attention):
    num_layers = len(attention)
    agg_att = attention[-1][0].max(0)[0] # R1xR2
    for i in range(num_layers-1):
        att = attention[num_layers-2-i][0].max(0)[0] # R1xR2
        agg_att = torch.max(agg_att,att)
        #agg_att = agg_att + att
    
    # mean = torch.mean(agg_att,0,keepdim=True)
    # std0 = torch.std(agg_att,0,keepdim=True)
    # std1 = torch.std(agg_att,1,keepdim=True)
    # agg_att = (agg_att - mean) / torch.sqrt(std0*std1)
    #agg_att = agg_att / num_layers
    #agg_att = agg_att / torch.sum(agg_att,1,keepdim=True)
        
    return agg_att


def compute_lang_guided_obj_obj_att(word_obj_att,word_word_att):
    word_word_att = word_word_att.pow(0.5)
    att = torch.sum(
        word_word_att.unsqueeze(3)*word_obj_att.unsqueeze(1),2) # BxWxO
    obj_word_att = word_obj_att.permute(0,2,1) # BxOxW
    att = torch.sum(obj_word_att.unsqueeze(3)*att.unsqueeze(1),2)
    # max_att = torch.max(att,2)[0].unsqueeze(2)
    # min_att = torch.min(att,2)[0].unsqueeze(2)
    # att = (att - min_att) / (max_att - min_att)
    return att


def eval_model(model,dataloader,exp_const):
    # Set mode
    model.object_encoder.eval()
    model.lang_sup_criterion.eval()

    num_samples = 0
    for it,data in enumerate(tqdm(dataloader)):
        # if data['image_name'][0]!='COCO_val2014_000000183677':
        #     continue

        if (exp_const.num_vis_samples is not None) and \
            (num_samples >= exp_const.num_vis_samples):
                break

        # Forward pass
        object_features = data['features'].cuda()
        object_mask = data['object_mask'].cuda()
        pad_mask = data['pad_mask'].cuda()
        context_object_features, obj_obj_att = model.object_encoder(
            object_features)
        obj_obj_att = agg_attention(obj_obj_att)

        if exp_const.vis_lang==True:
            token_ids, tokens, token_lens = model.cap_encoder.tokenize_batch(
                data['caption'])
            token_ids = torch.LongTensor(token_ids).cuda()
            word_features, word_word_att = model.cap_encoder(token_ids)
            word_word_att = torch.max(torch.cat(word_word_att,1),1)[0] # BxWxW
            noun_token_ids = data['noun_token_ids'].cuda()
            word_features, token_mask = model.cap_encoder.select_noun_embed(
                word_features,
                noun_token_ids)
            noun_noun_att = model.cap_encoder.select_noun_att(
                word_word_att,
                noun_token_ids)

            lang_sup_loss, noun_obj_att_ = model.lang_sup_criterion(
                context_object_features,
                object_features,
                word_features.detach(),
                token_mask)
            
            Bw,Bo,Tw,To = noun_obj_att_.size()
            noun_obj_att = torch.zeros([Bw,Tw,To],dtype=torch.float32).cuda()
            for b in range(Bw):
                noun_obj_att[b] = noun_obj_att_[b,b]

            lang_guided_obj_obj_att = compute_lang_guided_obj_obj_att(
                noun_obj_att,
                noun_noun_att)

            att = lang_guided_obj_obj_att[0]
        else:
            att = obj_obj_att
            
        num_objects = data['num_objects'][0].item()

        if num_objects >= 4:
            K = 4
        else:
            K = num_objects

        _, indices = torch.topk(att,K,1)
        box_ids = indices.detach().cpu().numpy()
        att = att.detach().cpu().numpy()
        query_att = np.transpose(np.max(att,0,keepdims=True))
        #att = att*query_att

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
        num_boxes = min(10,len(boxes))
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
            (0,0,255),
            (255,255,255)]
        for i in range(num_boxes):
            box_img = img
            box_img = vis_bbox(boxes[i],box_img,(255,255,0),modify=False)
            dst_filename = os.path.join(vis_dir,str(i)+'_query.jpg')
            skio.imsave(dst_filename,box_img)

            box_img = 0*img
            box_img = box_img.astype(np.float32)
            for k in range(K):
                if box_ids[i,k] >= len(boxes):
                    break
                
                box_img = create_att(
                    boxes[box_ids[i,k]],
                    box_img,
                    np.minimum(1,att[i,box_ids[i,k]]))
                #print(boxes[box_ids[i,k]],k,np.minimum(1,att[i,k]))
                
                # box_img = vis_bbox(
                #     boxes[box_ids[i,k]],
                #     box_img,
                #     colors[k54],
                #     modify=False)
            
            box_img = box_img*img.astype(np.float32)
            box_img = box_img.astype(np.uint8)

            for k in range(K):
                if box_ids[i,k] >= len(boxes):
                    break
                
                box_img = vis_bbox(
                    boxes[box_ids[i,k]],
                    box_img,
                    colors[k],
                    modify=False,
                    alpha=0)
            
            dst_filename = os.path.join(vis_dir,str(i)+'_att.jpg')
            skio.imsave(dst_filename,box_img)

            #query_att = np.mean(att[:,i])
            col_dict = {
                0: query_att[i,0],
                1: html_writer.image_tag(str(i)+'_query.jpg'),
                2: html_writer.image_tag(str(i)+'_att.jpg'),
                3: att[i,box_ids[i]],
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
        model.object_encoder.load_state_dict(
            torch.load(model.const.object_encoder_path)['state_dict'])
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

    with torch.no_grad():
        eval_model(model,dataloader,exp_const)