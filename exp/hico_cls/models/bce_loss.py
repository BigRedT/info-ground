import copy
import torch
import torch.nn as nn

import utils.io as io


class BCELoss(nn.Module,io.WritableToFile):
    def __init__(self):
        super().__init__()
        self.log_sigmoid = nn.LogSigmoid()

    def forward(self,logits,pos_labels,neg_labels):
        log_pos_prob = self.log_sigmoid(logits)
        log_neg_prob = log_pos_prob-logits
        pos_loss = -pos_labels*log_pos_prob
        neg_loss = -neg_labels*log_neg_prob
        num_labels = pos_labels.sum() + neg_labels.sum()
        #print(pos_labels.sum(),neg_labels.sum())
        loss = pos_loss.sum() + neg_loss.sum()
        loss = loss / (num_labels + 1e-6)
        return loss


class BalancedBCELoss(nn.Module,io.WritableToFile):
    def __init__(self):
        super().__init__()
        self.log_sigmoid = nn.LogSigmoid()

    def forward(self,logits,pos_labels,neg_labels):
        log_pos_prob = self.log_sigmoid(logits)
        log_neg_prob = log_pos_prob-logits
        num_pos = pos_labels.sum(0,keepdim=True)
        num_negs = neg_labels.sum(0,keepdim=True)
        pos_loss = -pos_labels*log_pos_prob / (num_pos + 1e-6)
        pos_loss = pos_loss.sum(0).mean()
        neg_loss = -neg_labels*log_neg_prob / (num_negs + 1e-6)
        neg_loss = neg_loss.sum(0).mean()
        loss = 0.5*(pos_loss + neg_loss)
        return loss