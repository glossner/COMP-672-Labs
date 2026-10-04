"""Week 5 student starter: compositional toy translation, GRU encoder-decoder."""
import math
import torch
from torch import nn
from torch.nn import functional as F
torch.set_num_threads(1)

def context_attention(query,keys):
    """query B,d; keys B,T,d; dot-product attention over source positions."""
    raise NotImplementedError("Complete this laboratory task.")

def beam_search(step,bos,eos,width=2,max_tokens=4):
    """step(prefix)->log probabilities. Carry completed hypotheses without expansion.
    Rank raw cumulative log probability; return best completed, else best live prefix.
    """
    raise NotImplementedError("Complete this laboratory task.")

class Translator(nn.Module):
    def __init__(self,vocab=12,d=24):
        super().__init__(); self.emb=nn.Embedding(vocab,d)
        self.encoder=nn.GRU(d,d,batch_first=True); self.cell=nn.GRUCell(2*d,d)
        self.head=nn.Linear(2*d,vocab)
    def encode(self,src):
        keys,h=self.encoder(self.emb(src)); return keys,h[0]
    def decode(self,token,h,keys):
        ctx,_=context_attention(h,keys); h=self.cell(torch.cat([self.emb(token),ctx],-1),h)
        ctx,_=context_attention(h,keys); return self.head(torch.cat([h,ctx],-1)),h
    def teacher_forced(self,src,target):
        keys,h=self.encode(src); outputs=[]
        for t in range(target.size(1)-1):
            logits,h=self.decode(target[:,t],h,keys); outputs.append(logits)
        return torch.stack(outputs,1)

def sequence_loss(logits,target):
    raise NotImplementedError("Complete this laboratory task.")

def data():
    # 0 BOS, 1 EOS; source color (2..5), shape (6..9); target shape then color.
    # This invented translation task tests reordering, not natural French proficiency.
    pairs=[(c,s) for c in range(2,6) for s in range(6,10)]
    train=[p for p in pairs if (p[0]+p[1])%4!=0]; test=[p for p in pairs if p not in train]
    def tensors(rows):
        return torch.tensor(rows),torch.tensor([[0,s,c,1] for c,s in rows])
    return tensors(train),tensors(test)

def checks():
    q=torch.zeros(1,2); keys=torch.tensor([[[1.,0.],[0.,2.]]])
    ctx,w=context_attention(q,keys); assert torch.allclose(ctx,torch.tensor([[.5,1.]]))
    # IDs: BOS0, EOS1, A2, B3, C4. Demonstrates greedy search error.
    def step(prefix):
        p=[1e-12]*5
        if len(prefix)==1: p[2]=.6; p[3]=.4
        elif prefix[-1]==2: p[1]=.4; p[4]=.6
        elif prefix[-1]==3: p[1]=.95; p[4]=.05
        else: p[1]=1.
        return [math.log(v) for v in p]
    assert beam_search(step,0,1,1,3)[0]==[0,2,4,1]
    assert beam_search(step,0,1,2,3)[0]==[0,3,1]

def experiment():
    torch.manual_seed(17); (src,tgt),(test,truth)=data(); model=Translator()
    opt=torch.optim.Adam(model.parameters(),lr=.01)
    initial=sequence_loss(model.teacher_forced(src,tgt),tgt).item()
    for _ in range(250):
        opt.zero_grad(); loss=sequence_loss(model.teacher_forced(src,tgt),tgt); loss.backward(); opt.step()
    model.eval(); rows=[]
    for sentence,gold in zip(test,truth):
        keys,h0=model.encode(sentence[None])
        def step(prefix):
            h=h0
            for token in prefix:
                logits,h=model.decode(torch.tensor([token]),h,keys)
            return F.log_softmax(logits[0],-1).detach().tolist()
        greedy,_=beam_search(step,0,1,1,4); beam,score=beam_search(step,0,1,3,4)
        rows.append({'source':sentence.tolist(),'gold':gold.tolist(),'greedy':greedy,'beam':beam,'beam_logp':score})
    return {'initial_train_loss':initial,'final_train_loss':loss.item(),'held_out_combinations':rows,'greedy_exact_match':sum(r['greedy']==r['gold'] for r in rows)/len(rows),'beam_exact_match':sum(r['beam']==r['gold'] for r in rows)/len(rows)}

if __name__=='__main__': checks(); print(experiment())
