import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils.io as io


class Identity(nn.Module,io.WritableToFile):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x


class InfoNCE(nn.Module,io.WritableToFile):
    def __init__(self,fx=None,fc=None):
        super().__init__()
        if fx is None:
            fx = Identity()
        
        if fc is None:
            fc = Identity()

        self.fx = fx
        self.fc = fc
        
    def forward(self,x,c,mask=None):
        """
        Input:
        :x: BxTxD1 non-contextualized features
        :c: BxTxD2 contextualized features
        :mask: BxT boolean matrix denoting which time stamps are masked
        """
        assert(x.size()[:2]==c.size()[:2]), 'x and c must have same B and T'
        
        B,T,D1 = x.size()
        B,T,D2 = c.size()

        if mask is not None:
            err_str = f'size of mask must be {B}x{T} but got {mask.size()}'
            assert(len(mask.size())==2), err_str
            
            T_mask = (mask.sum(0) > 0)
            x = x[:,T_mask]
            c = c[:,T_mask]
            T_ = x.size(1)
        else:
            T_ = T

        x = self.fx(x.view(-1,D1)).view(B,T_,-1)
        c = self.fc(c.view(-1,D2)).view(B,T_,-1)

        x = x.unsqueeze(1) # Bx1xT_xD
        c = c.unsqueeze(0) # 1xBxT_xD
        logits = torch.sum(x*c,3) # BxBxT
        log_softmax1 = F.log_softmax(logits,1) # Select context given feature
        log_softmax2 = F.log_softmax(logits,0) # Select feature given context
        avg_log_softmax = 0.5*(log_softmax1 + 0*log_softmax2)
        loss = -avg_log_softmax.mean(2).diag().mean()

        return loss


if __name__=='__main__':
    B,T,D = (10,5,7)
    x = torch.rand(B,T,D)
    c1 = x+5
    c2 = 0*torch.rand(B,T,D)
    fc = nn.Sequential(nn.Linear(7,7),nn.BatchNorm1d(7),nn.ReLU(),nn.Linear(7,7))
    fx = nn.Sequential(nn.Linear(7,7),nn.BatchNorm1d(7),nn.ReLU(),nn.Linear(7,7))
    info_nce_loss = InfoNCE(fc,fx)
    loss2,_,_ = info_nce_loss(x,c2)
    opt = optim.SGD(info_nce_loss.parameters(),lr=1e-1)
    for i in range(5000):
        loss1,_,_ = info_nce_loss(x,c1)
        if i%100==0:
            print('Iter:',i,'Loss:',loss1.item())
        opt.zero_grad()
        loss1.backward()
        opt.step()
    
    for i in range(5000):
        loss2,_,_ = info_nce_loss(x,c2)
        if i%100==0:
            print('Iter:',i,'Loss:',loss2.item())
        opt.zero_grad()
        loss2.backward()
        opt.step()
    
    print('x==c','->',loss1.item())
    print('x!=c','->',loss2.item())

            
                
        

