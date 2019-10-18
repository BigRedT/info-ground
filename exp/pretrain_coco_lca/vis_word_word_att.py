import os
import h5py
import math
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import itertools
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import skimage.io as skio

import utils.io as io
from utils.constants import save_constants, Constants
from .object_encoder import ObjectEncoder
from .cap_encoder import CapEncoder
from .dataset import DetFeatDataset
from .info_nce_loss import InfoNCE
#from .cap_info_nce_loss import KVLayer, CapInfoNCE
from .factored_cap_info_nce_loss import CapInfoNCE, KLayer, FLayer
from utils.bbox_utils import vis_bbox
from utils.html_writer import HtmlWriter


def vis_att(att,words,filename):
    fig,ax = plt.subplots()
    im = ax.imshow(att)
    ax.set_xticks(np.arange(len(words)))
    ax.set_yticks(np.arange(len(words)))
    ax.set_xticklabels(words)
    ax.set_yticklabels(words)
    plt.setp(
        ax.get_xticklabels(), 
        rotation=45, 
        ha="right",
        rotation_mode="anchor")

    att = np.round(att,1)
    for i in range(len(words)):
        for j in range(len(words)):
            if att[i,j]==0:
                continue
            text = ax.text(j,i,att[i,j],ha="center", va="center", color="w")

    fig.tight_layout()
    plt.savefig(filename)
    return im

def get_word_word_att(att):
    # att: [Bx12xWxW]*12
    return torch.max(torch.cat(att,1),1)[0]

def eval_model(model,dataloader,exp_const):
    # Set mode
    html_writer = HtmlWriter(os.path.join(exp_const.vis_dir,'vis.html'))
    num_samples = 0
    for it,data in enumerate(tqdm(dataloader)):
        if (exp_const.num_vis_samples is not None) and \
            (num_samples >= exp_const.num_vis_samples):
                break

        # Forward pass
        token_ids, tokens, token_lens = model.cap_encoder.tokenize_batch(
            data['caption'])
        token_ids = torch.LongTensor(token_ids).cuda()
        word_features,att = model.cap_encoder(token_ids)
        
        att = get_word_word_att(att)[0]
        att = att.detach().cpu().numpy()

        noun_token_ids = data['noun_token_ids'][0].detach().numpy()
        noun_token_ids = [i for i in noun_token_ids if i!=-1]
        att_mask = 0*att
        for i in noun_token_ids:
            for j in noun_token_ids:
                att_mask[i,j] = 1
        
        att = att*att_mask
        filename = os.path.join(exp_const.vis_dir,str(it) + '.png')
        att_im = vis_att(att,tokens[0],filename)
        col_dict = {
            0: html_writer.image_tag(str(it) + '.png',height=400,width=500)
        }
        html_writer.add_element(col_dict)
        
        num_samples += 1
    
    html_writer.close()


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
    model.cap_encoder = CapEncoder(model.const.cap_encoder)
    model.cap_encoder.cuda()

    print('Creating dataloader ...')
    dataloaders = {}
    dataset = DetFeatDataset(data_const)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1)

    eval_model(model,dataloader,exp_const)