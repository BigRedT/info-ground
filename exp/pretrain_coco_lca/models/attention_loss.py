import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils.io as io


class AttentionLoss(nn.Module,io.WritableToFile):
    def __init__(self):
        super().__init__()
        pass

    def forward(self,att,max_att):
        return torch.mean(torch.max(torch.FloatTensor([0]).cuda(),att-max_att))


            
                
        

