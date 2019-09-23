import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils.io as io


class KVIdentity(nn.Module,io.WritableToFile):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        K = x
        V = x
        return K,V


class CapInfoNCE(nn.Module,io.WritableToFile):
    def __init__(self,fo=None,fw=None):
        super().__init__()
        if fo is None:
            fo = KVIdentity()
        
        if fw is None:
            fw = KVIdentity()

        self.fo = fo
        self.fw = fw
        
    def forward(self,o,w):
        """
        Input:
        :o: BoxToxDo object features
        :w: BwxTwxDw caption word features
        """
        assert(o.size()[:1]==w.size()[:1]), 'Bo==Bw'
        
        Bo,To,Do = o.size()
        Bw,Tw,Dw = w.size()

        Ko,Vo = self.fo(o)
        Kw,Vw = self.fw(w)
        
        D = Kw.size(2)

        Kw = Kw.unsqueeze(1).unsqueeze(3) # Bwx1xTwx1xD
        Ko = Ko.unsqueeze(1) # Box1xToxD
        att = torch.sum(Kw*Ko,4,keepdim=True) # BwxBoxTwxTox1
        att = att / torch.sqrt(torch.tensor(D).float()) # BwxBoxTwxTox1
        att = F.softmax(att,3)

        Vo = Vo.unsqueeze(1) # Box1xToxD
        att_Vo = torch.sum(att*Vo,3) # BwxBoxTwxD
        
        Vw = Vw.unsqueeze(1) # Bwx1xTwxD
        logits = torch.sum(att_Vo*Vw,3) # BwxBoxTw
        log_softmax = F.log_softmax(logits,1) # Select image given word
        loss = -log_softmax.mean(2).diag().mean()

        return loss


class KVLayer(nn.Module,io.WritableToFile):
    def __init__(self,d_in,d_out):
        super().__init__()
        self.K_layer = nn.Linear(d_in,d_out)
        self.V_layer = nn.Linear(d_in,d_out)
    
    def forward(self,x):
        B,T,D = x.size()
        x = x.view(-1,D)
        K = self.K_layer(x).view(B,T,-1)
        V = self.V_layer(x).view(B,T,-1)
        return K,V


if __name__=='__main__':
    B,Tw,To,D = (100,5,5,7)
    o = torch.rand(B,To,D).cuda()
    w = o
    #w = torch.rand(B,Tw,D).cuda()

    fo = KVLayer(7,7)
    fw = KVLayer(7,7)
    info_nce_loss = CapInfoNCE(fo,fw).cuda()
    
    opt = optim.SGD(info_nce_loss.parameters(),lr=1)
    for i in range(5000):
        loss = info_nce_loss(o,w)
        if i%100==0:
            print('Iter:',i,'Loss:',loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    print(loss.item())

            
                
        

