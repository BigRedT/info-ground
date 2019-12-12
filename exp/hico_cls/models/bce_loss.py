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