import copy
import torch
import torch.nn as nn

from context_layer.transformer import ContextLayerConstants, ContextLayer
import utils.io as io
from .info_nce_loss import InfoNCE


class HOIClassifierConstants(io.JsonSerializableClass):
    def __init__(self):
        super().__init__()
        self.object_feature_dim = 768
        self.context_layer = ContextLayerConstants()
        self.context_layer.num_hidden_layers = 3
        self.context_layer.hidden_size = 768
        self.context_layer.intermediate_size = 768
        self.num_hois = 600
        

class Identity(nn.Module,io.WritableToFile):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class HOIClassifier(nn.Module,io.WritableToFile):
    def __init__(self,const):
        super().__init__()
        self.const = copy.deepcopy(const)
        self.input_layer = Identity()
        # self.input_layer = nn.Sequential(
        #     nn.Linear(
        #         self.const.object_feature_dim,
        #         self.const.context_layer.hidden_size),
        #     nn.BatchNorm1d(self.const.context_layer.hidden_size),
        #     nn.ReLU())
        self.context_layer = ContextLayer(self.const.context_layer)
        self.cls_layer = nn.Linear(
            self.const.context_layer.hidden_size,
            self.const.num_hois)
        self.sigmoid_layer = nn.Sigmoid()

        self.pad_feat = nn.Parameter(
            data=torch.Tensor(self.const.object_feature_dim),
            requires_grad=True)
        self.mask_feat = nn.Parameter(
            data=torch.Tensor(self.const.object_feature_dim),
            requires_grad=True)
        self.cls_feat = nn.Parameter(
            data=torch.Tensor(self.const.context_layer.hidden_size),
            requires_grad=True)
        self.pad_feat.data.uniform_(-0.1,0.1)
        self.mask_feat.data.uniform_(-0.1,0.1) 
        self.cls_feat.data.uniform_(-0.1,0.1) 
    
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
        # Mask or pad object features
        object_features = self.preprocess_object_features(
            object_features,
            object_mask,
            pad_mask)

        # Transform object features to match dimension of classifier input
        # Can be set to Identity
        B,T,D = object_features.size()
        transformer_input = self.input_layer(
            object_features.view(-1,D)).view(B,T,-1)
        
        # Concat a trainable [CLS] token vector at T=0 position 
        cls_feat = self.cls_feat.unsqueeze(0).unsqueeze(0).repeat([B,1,1])
        transformer_input = torch.cat((cls_feat,transformer_input),1)

        # Pass through transformer to get features
        context_layer_output = self.context_layer(transformer_input)
        object_context_features = context_layer_output['last_hidden_states']
        
        # Classify the output at T=0 position
        hoi_logits = self.cls_layer(object_context_features[:,0,:])

        attention = context_layer_output['attention']
        if attention is not None:
            return hoi_logits, object_context_features, attention 

         
        return hoi_logits, object_context_features

