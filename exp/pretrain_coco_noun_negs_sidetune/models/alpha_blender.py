import torch
import torch.nn as nn


class AlphaBlender(nn.Module):
    def __init__(self,alpha_logit_init=0.0):
        super().__init__()
        self.alpha_logit = nn.Parameter(torch.tensor(alpha_logit_init))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x,y):
        alpha = self.sigmoid(self.alpha_logit)
        return alpha*x + (1-alpha)*y