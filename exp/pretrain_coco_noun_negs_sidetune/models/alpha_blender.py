import torch
import torch.nn as nn


class AlphaBlender(nn.Module):
    def __init__(self,alpha_logit_init=2.0,drop_prob=0.2):
        super().__init__()
        self.alpha_logit = nn.Parameter(torch.tensor(alpha_logit_init))
        self.sigmoid = nn.Sigmoid()
        #self.dropout = nn.Dropout(p=drop_prob)
    
    def forward(self,x,y):
        alpha = self.sigmoid(self.alpha_logit)
        #y = self.dropout(y)
        return alpha*x + (1-alpha)*y