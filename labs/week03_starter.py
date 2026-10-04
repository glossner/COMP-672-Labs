"""Week 3 student starter: tiny transformer; no downloaded weights or corpora."""
import math
import torch
from torch import nn
from torch.nn import functional as F
torch.set_num_threads(1)

def attention(q,k,v,causal=True):
    """Inputs B,H,T,d. Return attended values and row-normalized weights."""
    raise NotImplementedError("Complete this laboratory task.")

def causal_loss(logits,tokens):
    """logits at t predict token t+1; all sequences are equal length."""
    raise NotImplementedError("Complete this laboratory task.")

def masked_loss(logits,targets,selected):
    """selected is a Boolean B,T mask. No loss on unselected positions."""
    raise NotImplementedError("Complete this laboratory task.")

class TinyTransformer(nn.Module):
    def __init__(self,vocab=12,width=24,heads=3):
        super().__init__(); self.heads=heads; self.width=width
        self.embed=nn.Embedding(vocab,width); self.pos=nn.Embedding(12,width)
        self.qkv=nn.Linear(width,3*width); self.proj=nn.Linear(width,width)
        self.norm1=nn.LayerNorm(width); self.norm2=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,48),nn.ReLU(),nn.Linear(48,width))
        self.head=nn.Linear(width,vocab)
    def forward(self,ids,causal=True):
        b,t=ids.shape; d=self.width//self.heads
        x=self.embed(ids)+self.pos(torch.arange(t,device=ids.device))
        q,k,v=self.qkv(self.norm1(x)).chunk(3,-1)
        q,k,v=[z.view(b,t,self.heads,d).transpose(1,2) for z in (q,k,v)]
        z,_=attention(q,k,v,causal)
        x=x+self.proj(z.transpose(1,2).contiguous().view(b,t,self.width))
        x=x+self.ff(self.norm2(x)); return self.head(x)

class LoRALinear(nn.Module):
    def __init__(self,base,rank=2,alpha=2):
        super().__init__(); self.base=base; self.scale=alpha/rank
        for p in base.parameters(): p.requires_grad=False
        self.A=nn.Parameter(torch.randn(rank,base.in_features)*.02)
        self.B=nn.Parameter(torch.zeros(base.out_features,rank))
    def forward(self,x):
        raise NotImplementedError("Complete this laboratory task.")

def data(n,seed,shift=0):
    g=torch.Generator().manual_seed(seed)
    a=torch.randint(1,5,(n,),generator=g); b=torch.randint(5,9,(n,),generator=g)
    # 0 BOS, 9 separator, 10 EOS, 11 MASK; deterministic repeated structure.
    return torch.stack([a*0,a,b,a*0+9,((a-1+shift)%4)+1,b,a*0+10],1)

def checks():
    torch.manual_seed(17)
    q=torch.randn(2,3,4,8); k=torch.randn_like(q); v=torch.randn_like(q)
    z,w=attention(q,k,v,True)
    assert torch.allclose(w.sum(-1),torch.ones_like(w.sum(-1)))
    assert torch.count_nonzero(w.triu(1))==0
    changed=v.clone(); changed[:,:,3]+=100
    assert torch.allclose(attention(q,k,changed,True)[0][:,:,:3],z[:,:,:3])
    base=nn.Linear(4,3); layer=LoRALinear(base)
    x=torch.randn(2,4); assert torch.allclose(layer(x),base(x))
    logits=torch.zeros(1,3,12); selected=torch.tensor([[False,True,False]])
    assert torch.isclose(masked_loss(logits,torch.tensor([[0,1,2]]),selected),torch.tensor(math.log(12)))

def experiment():
    torch.manual_seed(17); train=data(128,17); dev=data(64,18); test=data(64,19)
    # Samples are independently generated draws of a finite grammar, not a claim of novel grammar generalization.
    model=TinyTransformer(); opt=torch.optim.Adam(model.parameters(),lr=.01)
    initial=causal_loss(model(dev),dev).item()
    for _ in range(100):
        opt.zero_grad(); loss=causal_loss(model(train),train); loss.backward(); opt.step()
    model.eval(); base_test=causal_loss(model(test),test).item()
    adapted=data(64,20,shift=1)
    for p in model.parameters(): p.requires_grad=False
    model.head=LoRALinear(model.head)
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.Adam(params,lr=.03)
    adapted_test=data(64,21,shift=1)
    before=causal_loss(model(adapted_test),adapted_test).item()
    for _ in range(80):
        opt.zero_grad(); loss=causal_loss(model(adapted),adapted); loss.backward(); opt.step()
    after=causal_loss(model(adapted_test),adapted_test).item()
    mlm=TinyTransformer(); opt=torch.optim.Adam(mlm.parameters(),lr=.01)
    selected=torch.zeros_like(train,dtype=torch.bool); selected[:,4]=True
    corrupt=train.clone(); corrupt[selected]=11
    for _ in range(80):
        opt.zero_grad(); loss=masked_loss(mlm(corrupt,False),train,selected); loss.backward(); opt.step()
    ts=torch.zeros_like(test,dtype=torch.bool); ts[:,4]=True; tc=test.clone(); tc[ts]=11
    return {'initial_dev_causal_loss':initial,'base_test_causal_loss':base_test,'adapter_trainable_parameters':sum(p.numel() for p in params),'adapted_distribution_loss_before':before,'adapted_test_loss_after':after,'masked_test_loss':masked_loss(mlm(tc,False),test,ts).item(),'warning':'Causal and masked losses score different events; do not rank their model quality from these numbers.'}

if __name__=='__main__':
    checks(); print(experiment())
