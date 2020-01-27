import torch
import torch.nn as nn
from torchvision.ops import roi_align


from .resnet import resnet34

class SideNet(nn.Module):
    def __init__(self,out_dim=1024):
        super().__init__()
        self.out_dim = out_dim
        self.cnn = resnet34(pretrained=True)
        self.fc = nn.Linear(512,self.out_dim)

    def forward(self,x,list_boxes):
        w1 = x.size(-1) # 224
        x = self.cnn(x,layer=5)
        w2 = x.size(-1) # 28
        
        x = roi_align(x,list_boxes,(7,7)).mean(3).mean(2)
        x = self.fc(x)
        
        roi_pooled_feats = []
        i = 0
        for boxes in list_boxes:
            n = boxes.size(0)
            roi_pooled_feats.append(x[i:i+n,:])
            i = i+n

        return roi_pooled_feats

    def pad_and_concat(self,feats,n_max=30):
        for i in range(len(feats)):
            n = feats[i].size(0)
            if n==n_max:
                continue
            elif n > n_max:
                feats[i] = feats[i][:n_max]
            else:
                D = feats[i].size(1)
                feats[i] = torch.cat((
                    feats[i],
                    torch.zeros([n_max-n,D],dtype=torch.float32).cuda()),0)
        
        feats = torch.stack(feats)
        return feats
            


if __name__=='__main__':
    net = SideNet()
    B=2
    C=3
    H=224
    W=224
    x = torch.zeros([B,C,H,W])
    x = net(x)
    import pdb; pdb.set_trace()