import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils.io as io


def compute_neg_noun_loss(
        att_V_o,
        noun_feats,
        valid_noun_mask,
        f_layer):
    """
    att_V_o: Bx(N+1)xD
    noun_feats: Bx(N+1)xDw
    valid_noun_mask: Bx1 (1s where valid 0s otherwise)
    f_layer: linear layer to compute V for words from CapInfoNCE. Can be 
        obtained using cap_info_nce.fw.f_layer
    """
    B,Np1,Dw = noun_feats.size()
    Vw = f_layer(noun_feats.view(-1,Dw)).view(B,Np1,-1) # Bx(N+1)xD
    
    #att_V_o_ = att_V_o[:,0].unsqueeze(1) # Bx1xD
    logits = torch.sum(att_V_o*Vw,2) # Bx(N+1)
    log_softmax = F.log_softmax(logits,1)
    log_softmax = valid_noun_mask*log_softmax
    num_valid = valid_noun_mask.sum()
    loss = -torch.sum(log_softmax[:,0]) / (num_valid + 1e-6)

    return loss, log_softmax