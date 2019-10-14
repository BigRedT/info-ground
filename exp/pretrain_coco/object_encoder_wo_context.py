import copy
import torch
import torch.nn as nn

import utils.io as io
from utils.constants import Constants
from .info_nce_loss import InfoNCE


class ResidualBlock(nn.Module,io.WritableToFile):
    def __init__(self,h):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(h,h),
            nn.ReLU(),
            nn.Dropout(0.1))
        self.layer_norm = nn.LayerNorm(h)
        
    def forward(self,x):
        return self.layer_norm(x + self.dense(x))
    

class ObjectEncoderConstants(io.JsonSerializableClass):
    def __init__(self):
        super().__init__()
        self.object_feature_dim = 1024
        self.context_layer = Constants()
        self.context_layer.hidden_size = 768

class ObjectEncoder(nn.Module,io.WritableToFile):
    def __init__(self,const):
        super().__init__()
        self.const = copy.deepcopy(const)
        self.input_layer = nn.Sequential(
            nn.Linear(
                self.const.object_feature_dim,
                self.const.context_layer.hidden_size),
            nn.BatchNorm1d(self.const.context_layer.hidden_size),
            nn.ReLU())
        self.context_layer = nn.Sequential(
            ResidualBlock(self.const.context_layer.hidden_size),
            ResidualBlock(self.const.context_layer.hidden_size),
            ResidualBlock(self.const.context_layer.hidden_size),
            ResidualBlock(self.const.context_layer.hidden_size))
        self.pad_feat = nn.Parameter(
            data=torch.Tensor(self.const.object_feature_dim),
            requires_grad=True)
        self.mask_feat = nn.Parameter(
            data=torch.Tensor(self.const.object_feature_dim),
            requires_grad=True)
        self.pad_feat.data.uniform_(-0.1,0.1)
        self.mask_feat.data.uniform_(-0.1,0.1) 
    
    

    def preprocess_object_features(self,object_features,object_mask,pad_mask):
        if object_mask is not None:
            object_mask = object_mask.unsqueeze(2).float()
            object_features = object_mask*self.mask_feat + \
                (1-object_mask)*object_features

        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(2).float()
            object_features = pad_mask*self.pad_feat + \
                (1-pad_mask)*object_features

        return object_features

    def forward(self,object_features,object_mask=None,pad_mask=None):
        object_features = self.preprocess_object_features(
            object_features,
            object_mask,
            pad_mask)

        B,T,D = object_features.size()
        transformer_input = self.input_layer(object_features.view(-1,D))
        object_context_features = self.context_layer(
            transformer_input).view(B,T,-1)

        return object_context_features

