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
    def __init__(self,fo=None,fw=None,ku=None,kw=None):
        super().__init__()
        self.fo = fo
        self.fw = fw
        self.ku = ku
        self.kw = kw
        
    def forward(self,o,u,w,mask):
        """
        Input:
        :o: BoxToxDo contextualized object features
        :u: BoxToxDu uncontextualized object features
        :w: BwxTwxDw caption word features
        :mask: BwxTw word mask
        """
        assert(o.size()[:1]==w.size()[:1]), 'Bo==Bw'
        
        Bo,To,Do = o.size()
        _,_,Du = u.size()
        Bw,Tw,Dw = w.size()

        Ku = self.ku(u)
        Kw = self.kw(w)
        
        D = Kw.size(2)

        Kw = Kw.unsqueeze(1).unsqueeze(3) # Bwx1xTwx1xD
        Ku = Ku.unsqueeze(1) # Box1xToxD
        att = torch.sum(Kw*Ku,4,keepdim=True) # BwxBoxTwxTox1
        att = att / torch.sqrt(torch.tensor(D).float()) # BwxBoxTwxTox1
        att = F.softmax(att,3)
        
        o = o.unsqueeze(1) # Box1xToxDo
        V_o = self.fo(o)
        att_V_o = torch.sum(att*V_o,3) # BwxBoxTwxDo
        
        w = w.unsqueeze(1) # Bwx1xTwxDw
        V_w = self.fw(w) # Bwx1xTwxD

        logits = torch.sum(att_V_o*V_w,3) # BwxBoxTw
        log_softmax = F.log_softmax(logits,1) # Select image given word
        mask = mask.unsqueeze(1) # Bwx1xTw
        log_softmax = (1-mask)*log_softmax
        num_non_mask = torch.sum(1-mask,2,keepdim=True) # Bwx1x1
        log_softmax = log_softmax / (num_non_mask + 1e-6)
        loss = -log_softmax.sum(2).diag().mean()

        att = att.squeeze(4)
        
        return loss, att, att_V_o

    def att_V_o(self,o,u,w):
        """
        Input:
        :o: BoxToxDo contextualized object features
        :u: BoxToxDu uncontextualized object features
        :w: BwxTwxDw caption word features
        """
        assert(o.size()[:1]==w.size()[:1]), 'Bo==Bw'
        
        Bo,To,Do = o.size()
        _,_,Du = u.size()
        Bw,Tw,Dw = w.size()

        Ku = self.ku(u)
        Kw = self.kw(w)
        
        D = Kw.size(2)

        Kw = Kw.unsqueeze(1).unsqueeze(3) # Bwx1xTwx1xD
        Ku = Ku.unsqueeze(1) # Box1xToxD
        att = torch.sum(Kw*Ku,4,keepdim=True) # BwxBoxTwxTox1
        att = att / torch.sqrt(torch.tensor(D).float()) # BwxBoxTwxTox1
        att = F.softmax(att,3)
        
        o = o.unsqueeze(1) # Box1xToxDo
        V_o = self.fo(o) # Box1xToxD
        att_V_o = torch.sum(att*V_o,3) # BwxBoxTwxD

        return att_V_o

    def att_V_o_for_verbs(self,o,u,w):
        """
        Input:
        :o: BoxToxDo contextualized object features
        :u: BoxToxDu uncontextualized object features
        :w: BwxTwxDw caption word features
        """
        assert(o.size()[:1]==w.size()[:1]), 'Bo==Bw'
        
        Bo,To,Do = o.size()
        _,_,Du = u.size()
        Bw,Tw,Dw = w.size()

        Ku = self.ku(u)
        Kw = self.kw(w)
        
        D = Kw.size(2)

        Kw = Kw.unsqueeze(2) # BwxTwx1xD
        Ku = Ku.unsqueeze(1) # Box1xToxD
        att = torch.sum(Kw*Ku,3,keepdim=True) # BxTwxTox1
        att = att / torch.sqrt(torch.tensor(D).float()) # BxTwxTox1
        att = F.softmax(att,2)
        
        o = o.unsqueeze(1) # Box1xToxDo
        V_o = self.fo(o) # Box1xToxD
        att_V_o = torch.sum(att*V_o,2) # BxTwxD

        return att_V_o

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


class KLayer(nn.Module,io.WritableToFile):
    def __init__(self,d_in,d_out):
        super().__init__()
        self.K_layer = nn.Linear(d_in,d_out)
    
    def forward(self,x):
        B,T,D = x.size()
        x = x.view(-1,D)
        K = self.K_layer(x).view(B,T,-1)
        return K


class FLayer(nn.Module,io.WritableToFile):
    def __init__(self,d_in,d_out):
        super().__init__()
        self.f_layer = nn.Linear(d_in,d_out)
    
    def forward(self,x):
        Bw,Bo,Tw,Do = x.size()
        x = x.view(-1,Do)
        x = self.f_layer(x)
        x = x.view(Bw,Bo,Tw,-1)
        return x

class FLayer3d(nn.Module,io.WritableToFile):
    def __init__(self,d_in,d_out):
        super().__init__()
        self.f_layer = nn.Linear(d_in,d_out)
    
    def forward(self,x):
        Bw,Bo,Tw,Do = x.size()
        x = x.view(-1,Do)
        x = self.f_layer(x)
        x = x.view(Bw,Bo,Tw,-1)
        return x


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

            
                
        

