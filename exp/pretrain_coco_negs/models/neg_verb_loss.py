import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils.io as io


def compute_neg_verb_loss(
        att_V_o,
        verb_ids,
        neg_verb_feats,
        pos_verb_feats,
        f_layer):
    """
    att_V_o: BwxBoxTwxDo
    verb_ids: B
    neg_verb_feats: BxNxD
    pos_verb_feats: BxD
    f_layer: linear layer to compute V for words from CapInfoNCE. Can be 
        obtained using cap_info_nce.fw.f_layer
    """
    pos_verb_feats = pos_verb_feats.unsqueeze(1) # Bx1xD
    verb_feats = torch.cat((pos_verb_feats,neg_verb_feats),1) # BxN+1xD
    B,Np1,D = verb_feats.size()
    Vw_ = f_layer(verb_feats.view(-1,D)).view(B,Np1,-1)

    loss = 0
    Vo = []
    Vw = []
    for i in range(B):
        verb_id = verb_ids[i]
        if verb_id==-1:
            continue

        Vo.append(att_V_o[i,i,verb_id]) # list of Do dim vectors
        Vw.append(Vw_[i])   # list of (N+1)xDo dim matrices

    if len(Vo)==0:
        return 0

    Vo = torch.stack(Vo,0) # BxDo
    Vo = Vo.view(B,1,-1) # Bx1xDo

    Vw = torch.stack(Vw,0) # Bx(N+1)xDo
    
    logits = torch.sum(Vo*Vw,2) # Bx(N+1)
    log_softmax = F.log_softmax(logits,1)
    loss = -torch.mean(log_softmax[:,0])

    return loss, log_softmax



        
        

    
