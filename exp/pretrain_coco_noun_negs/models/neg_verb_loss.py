import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils.io as io


def compute_neg_verb_loss(
        att_V_o,
        verb_feats,
        valid_verb_mask,
        f_layer):
    """
    att_V_o: Bx(N+1)xD
    verb_feats: Bx(N+1)xDw
    valid_verb_mask: Bx1 (1s where valid 0s otherwise)
    f_layer: linear layer to compute V for words from CapInfoNCE. Can be 
        obtained using cap_info_nce.fw.f_layer
    """
    B,Np1,Dw = verb_feats.size()
    Vw = f_layer(verb_feats.view(-1,Dw)).view(B,Np1,-1) # Bx(N+1)xD
    #Vw = Vw.detach() # We want gradient only for attended object representation
    
    #att_V_o_ = att_V_o[:,0].unsqueeze(1) # Bx1xD
    logits = torch.sum(att_V_o*Vw,2) # Bx(N+1)
    log_softmax = F.log_softmax(logits,1)
    log_softmax = valid_verb_mask*log_softmax
    num_valid = valid_verb_mask.sum()
    loss = -torch.sum(log_softmax[:,0]) / (num_valid + 1e-6)

    return loss, log_softmax


# def compute_neg_verb_loss(
#         att_V_o,
#         verb_ids,
#         neg_verb_feats,
#         gt_token_feats,
#         f_layer):
#     """
#     att_V_o: BwxBox1xDo
#     verb_ids: B
#     neg_verb_feats: BxNxD
#     gt_token_feats: BxTxD
#     f_layer: linear layer to compute V for words from CapInfoNCE. Can be 
#         obtained using cap_info_nce.fw.f_layer
#     """
#     B,T,D = gt_token_feats.size()
#     pos_verb_feats_ = []
#     neg_verb_feats_ = []
#     att_V_o_ = []
#     for i in range(B):
#         verb_id = verb_ids[i].item()
#         if verb_id==-1 or verb_id >= T:
#             continue
#         pos_verb_feats_.append(gt_token_feats[i,verb_id].unsqueeze(0)) #1xD
#         neg_verb_feats_.append(neg_verb_feats[i]) #NxD
#         att_V_o_.append(att_V_o[i,i,0]) # 1xD

#     if len(att_V_o_)==0:
#         return torch.tensor(0).cuda(),None

#     verb_feats = torch.cat((
#         torch.stack(pos_verb_feats_,0),
#         torch.stack(neg_verb_feats_,0)),1) # Bx(N+1)xD
#     B,Np1,D = verb_feats.size()
#     Vw = f_layer(verb_feats.view(-1,D)).view(B,Np1,-1) # Bx(N+1)xDo
#     Vw = Vw.detach() # We want gradient only for attended object representation

#     att_V_o_ = torch.stack(att_V_o_,0) # BxDo
#     att_V_o_ = att_V_o_.view(B,1,-1) # Bx1xDo
    
#     logits = torch.sum(att_V_o_*Vw,2) # Bx(N+1)
#     log_softmax = F.log_softmax(logits,1)
#     loss = -torch.mean(log_softmax[:,0])

#     return loss, log_softmax


# def compute_neg_verb_loss(
#         att_V_o,
#         verb_ids,
#         neg_verb_feats,
#         gt_token_feats,
#         f_layer):
#     """
#     att_V_o: BwxBoxTwxDo
#     verb_ids: B
#     neg_verb_feats: BxNxD
#     gt_token_feats: BxTxD
#     f_layer: linear layer to compute V for words from CapInfoNCE. Can be 
#         obtained using cap_info_nce.fw.f_layer
#     """
#     B,T,D = gt_token_feats.size()
#     pos_verb_feats_ = []
#     neg_verb_feats_ = []
#     Vo = []
#     for i in range(B):
#         verb_id = verb_ids[i].item()
#         if verb_id==-1 or verb_id >= T:
#             continue
#         pos_verb_feats_.append(gt_token_feats[i,verb_id].unsqueeze(1)) #BxD
#         neg_verb_feats_.append(neg_verb_feats[i]) #BxNxD
#         try:
#             Vo.append(att_V_o[i,i,verb_id]) # Bx(N+1)xD
#         except IndexError:
#             import pdb; pdb.set_trace()

#     if len(Vo)==0:
#         return torch.tensor(0).cuda(),None

#     verb_feats = torch.cat((
#         torch.cat(pos_verb_feats_,0),
#         torch.cat(neg_verb_feats_,0)),1) # Bx(N+1)xD
#     B,Np1,D = verb_feats.size()
#     Vw = f_layer(verb_feats.view(-1,D)).view(B,Np1,-1) # Bx(N+1)xDo

#     Vo = torch.stack(Vo,0) # BxDo
#     Vo = Vo.view(B,1,-1) # Bx1xDo

#     Vw = torch.stack(Vw,0) # Bx(N+1)xDo
    
#     logits = torch.sum(Vo*Vw,2) # Bx(N+1)
#     log_softmax = F.log_softmax(logits,1)
#     loss = -torch.mean(log_softmax[:,0])

#     return loss, log_softmax
        

    
