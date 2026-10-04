"""Week 1 student starter. Original instructional corpus."""
from collections import Counter, defaultdict
import math
import re
import unicodedata
import numpy as np

def tokenize(text):
    """NFC, lower case, Unicode word sequences and individual punctuation."""
    raise NotImplementedError("Complete this laboratory task.")

def learn_bpe(words, n_merges=12):
    """Word->count input; character initialization; no word boundary symbol.
    Return an ordered list of merged pairs. Use lexical tie breaking.
    """
    raise NotImplementedError("Complete this laboratory task.")

def apply_bpe(word, merges):
    symbols=list(word)
    for a,b in merges:
        out=[]; i=0
        while i<len(symbols):
            if i+1<len(symbols) and symbols[i]==a and symbols[i+1]==b:
                out.append(a+b); i+=2
            else: out.append(symbols[i]); i+=1
        symbols=out
    return symbols

def corpus():
    subjects=['robot','student','engineer','teacher','doctor','pilot','artist','researcher']
    verbs=['reads','checks','finds','moves','tests','describes']
    objects=['a signal','the report','a token','the sample']
    texts=[f'the {s} {v} {o} .' for s in subjects for v in verbs for o in objects]
    rng=np.random.default_rng(17); rng.shuffle(texts)
    return texts[:120],texts[120:156],texts[156:]

class Bigram:
    def __init__(self, train, alpha=1.0):
        if alpha<=0: raise ValueError('alpha must be positive')
        self.vocab=sorted(set(t for s in train for t in tokenize(s))|{'<UNK>','<EOS>'})
        self.vset=set(self.vocab); self.alpha=alpha
        self.counts=Counter(); self.hist=Counter()
        for text in train:
            seq=['<BOS>']+tokenize(text)+['<EOS>']
            for a,b in zip(seq,seq[1:]): self.counts[a,b]+=1; self.hist[a]+=1
    def probability(self, history, token):
        raise NotImplementedError("Complete this laboratory task.")
    def perplexity(self, texts):
        """Include EOS, exclude BOS from scored-event count; map OOV to UNK."""
        raise NotImplementedError("Complete this laboratory task.")

def checks():
    assert tokenize('café')==tokenize('cafe\u0301')
    assert learn_bpe({'low':2,'lower':1},2)==[('l','o'),('lo','w')]
    m=Bigram(['a b','a c'])
    for h in ['a','<BOS>','unseen']:
        assert np.isclose(sum(m.probability(h,t) for t in m.vocab),1)
    assert math.isfinite(m.perplexity(['new unseen words']))

def experiment():
    train,dev,test=corpus()
    results={a:Bigram(train,a).perplexity(dev) for a in [.01,.1,1.]}
    best=min(results,key=results.get)
    words=Counter(t for s in train for t in tokenize(s))
    return {'dev_perplexities':results,'chosen_alpha':best,'test_perplexity':Bigram(train,best).perplexity(test),'bpe_merges':learn_bpe(words,12),'test_documents':len(test)}

if __name__=='__main__':
    checks(); print(experiment())
