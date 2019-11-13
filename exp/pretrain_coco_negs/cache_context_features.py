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

import utils.io as io
from utils.constants import save_constants, Constants
from .models.object_encoder import ObjectEncoder


def cache(f,context_f,model):
    images_wo_objects = []
    for image_id in tqdm(f.keys()):
        object_features = torch.cuda.FloatTensor(f[image_id][()]).unsqueeze(0)
        
        if object_features.size(1) == 0:
            images_wo_objects.append(image_id)
            continue

        context_features = model.object_encoder(object_features)    
        context_features = context_features[0]
        context_f.create_dataset(
            image_id,
            data=context_features.cpu().detach().numpy())

    print(len(images_wo_objects))
    

def main(exp_const,data_const,model_const):
    os.environ['HDF5_USE_FILE_LOCKING']="FALSE"

    print('Loading model ...')
    model = Constants()
    model.const = model_const
    model.object_encoder = ObjectEncoder(model.const.object_encoder)
    loaded_object = torch.load(model.const.object_encoder_path)
    model.object_encoder.load_state_dict(loaded_object['state_dict'])
    model.object_encoder.cuda()
    model.object_encoder.eval()
    step = loaded_object['step']
    print('\tModel loaded from step:',step)

    print('Loading features hdf5 file ...')
    f = io.load_h5py_object(data_const.features_hdf5)

    print('Creating context features hdf5 file ...')
    context_f = h5py.File(data_const.context_features_hdf5,'w')

    with torch.no_grad():
        cache(f,context_f,model)
    
    f.close()
    context_f.close()