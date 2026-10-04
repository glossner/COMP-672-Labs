"""Week 7 student starter. Tiny residual quantizer and simulated rating analysis."""
import math
import numpy as np

def residual_quantize(x,codebooks):
    """x N,d; each codebook K,d. Return N,Q indices and N,d reconstruction."""
    raise NotImplementedError("Complete this laboratory task.")

def nominal_bitrate(frames_per_second,entries_per_codebook):
    """Fixed-length code indices, no entropy coding or transport overhead."""
    raise NotImplementedError("Complete this laboratory task.")

def paired_sentence_bootstrap(a,b,iterations=1000,seed=17):
    """a,b shape sentence,rater. Resample sentences, keeping raters fixed.
    CI covers sentence sampling only; not generalization to a population of raters.
    """
    raise NotImplementedError("Complete this laboratory task.")

def checks():
    books=[np.array([[0.],[.5],[1.]]),np.array([[-.25],[0.],[.25]])]
    ix,reconstruction=residual_quantize(np.array([[.7]]),books)
    assert ix.tolist()==[[1,2]] and np.isclose(reconstruction[0,0],.75)
    assert nominal_bitrate(50,[1024]*8)==4000
    # Includes non-power-of-two sizes so the fixed-length coding convention is explicit.
    assert nominal_bitrate(10,[3])==20
    x=np.arange(12).reshape(4,3)
    result=paired_sentence_bootstrap(x,x)
    assert result['mean_difference']==0 and result['ci95']==[0.,0.]

def experiment():
    rng=np.random.default_rng(17); x=rng.uniform(-1,1,(200,1))
    books=[np.linspace(-1,1,8)[:,None],np.linspace(-.15,.15,8)[:,None]]
    one=residual_quantize(x,books[:1])[1]; two=residual_quantize(x,books)[1]
    # Simulated ratings exercise the analysis pipeline. These are not human listening results.
    a=np.clip(rng.normal(3.8,.5,(24,4)),1,5); b=np.clip(a-rng.normal(.2,.3,(24,4)),1,5)
    return {'one_stage_mse':float(np.mean((x-one)**2)),'two_stage_mse':float(np.mean((x-two)**2)),'one_stage_bits_per_second':nominal_bitrate(50,[8]),'two_stage_bits_per_second':nominal_bitrate(50,[8,8]),'simulated_sentence_bootstrap':paired_sentence_bootstrap(a,b),'note':'Scalar quantization is a teaching analogue of residual VQ, not a trained neural speech codec or a TTS synthesizer.'}

if __name__=='__main__': checks(); print(experiment())
