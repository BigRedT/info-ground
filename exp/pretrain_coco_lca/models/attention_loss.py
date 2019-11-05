import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils.io as io


# class AttentionLoss(nn.Module,io.WritableToFile):
#     def __init__(self):
#         super().__init__()
#         pass

#     def forward(self,att,max_att):
#         return torch.mean(torch.max(torch.FloatTensor([0]).cuda(),att-max_att))


# class AttentionLoss(nn.Module,io.WritableToFile):
#     def __init__(self):
#         super().__init__()
#         pass

#     def forward(self,att,lang_att):
#         """
#         att: BxHxRxR
#         lang_att: BxRxR
#         """
#         lang_att = torch.min(torch.tensor(1).float().cuda(),lang_att)
#         lang_att = lang_att.unsqueeze(1)
#         att = torch.min(torch.tensor(1).float().cuda(),att)
#         import pdb; pdb.set_trace()
#         # p1 = att
#         # p2 = lang_att
#         p1 = lang_att
#         p2 = att
#         log_p1 = torch.log(p1+1e-6)
#         log_p2 = torch.log(p2+1e-6)
#         log_1_p1 = torch.log(1-p1+1e-6)
#         log_1_p2 = torch.log(1-p2+1e-6)
#         loss = torch.mean(p1*(log_p1-log_p2) + (1-p1)*(log_1_p1-log_1_p2))
#         return loss
        #import pdb; pdb.set_trace()


class AttentionLoss(nn.Module,io.WritableToFile):
    def __init__(self):
        super().__init__()
        pass

    def forward(self,att,lang_att):
        """
        att: BxHxRxR
        lang_att: BxRxR
        """
        lang_att = torch.min(torch.tensor(1).float().cuda(),lang_att)
        lang_att = lang_att.unsqueeze(1)
        att = torch.min(torch.tensor(1).float().cuda(),att)
        p1 = lang_att
        p2 = att
        log_p1 = torch.log(p1+1e-6)
        log_p2 = torch.log(p2+1e-6)
        loss = torch.mean(torch.max(torch.tensor(0).float().cuda(),0.01+p2-p1))
        return loss
                
        

