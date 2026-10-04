"""Week 6 student starter. Synthesized tones demonstrate DSP; supplied transcripts test WER."""
import itertools
import numpy as np

def frame_signal(x,frame=400,hop=160):
    """No centering or padding. Reject signals shorter than one frame."""
    raise NotImplementedError("Complete this laboratory task.")

def mel_filters(sample_rate=16000,n_fft=512,bands=40):
    hz_to_mel=lambda f:2595*np.log10(1+f/700)
    mel_to_hz=lambda m:700*(10**(m/2595)-1)
    edges=mel_to_hz(np.linspace(hz_to_mel(0),hz_to_mel(sample_rate/2),bands+2))
    bins=np.arange(n_fft//2+1)*sample_rate/n_fft
    return np.array([np.maximum(0,np.minimum((bins-a)/(b-a),(c-bins)/(c-b))) for a,b,c in zip(edges,edges[1:],edges[2:])])

def log_mel(x,sample_rate=16000,frame=400,hop=160,n_fft=512,bands=40):
    raise NotImplementedError("Complete this laboratory task.")

def ctc_collapse(path,blank=0):
    raise NotImplementedError("Complete this laboratory task.")

def ctc_probability(frame_probs,target,blank=0):
    """Exact enumeration for small examples only: exponential in frame count."""
    total=0.
    for path in itertools.product(range(frame_probs.shape[1]),repeat=len(frame_probs)):
        if ctc_collapse(path,blank)==tuple(target):
            total+=float(np.prod([frame_probs[t,v] for t,v in enumerate(path)]))
    return total

def word_errors(reference,hypothesis):
    """Whitespace tokenization; unit costs. Tie preference: match, substitution, deletion, insertion.
    Return S,D,I,N and WER. An empty reference raises to avoid an undefined denominator.
    """
    raise NotImplementedError("Complete this laboratory task.")

def checks():
    assert frame_signal(np.zeros(16000)).shape==(98,400)
    assert log_mel(np.zeros(16000)).shape==(98,40)
    assert np.isfinite(log_mel(np.zeros(16000))).all()
    assert ctc_collapse([1,1,0,1,2,2,0])==(1,1,2)
    assert np.isclose(ctc_probability(np.array([[.4,.6],[.4,.6]]),[1]),.84)
    assert word_errors('we need a blue car','we need blue cars')['WER']==.4
    assert word_errors('a','a b c')['WER']==2.

def experiment():
    rng=np.random.default_rng(17); t=np.arange(16000)/16000
    x=.6*np.sin(2*np.pi*220*t)+.2*np.sin(2*np.pi*440*t)
    clean=log_mel(x); noisy=log_mel(x+.15*rng.normal(size=len(x)))
    refs=['we need a blue car','the radio is ready','turn left at the gate']
    hypotheses=['we need blue cars','the radio ready','turn right at the gate']
    rows=[word_errors(r,h) for r,h in zip(refs,hypotheses)]
    return {'log_mel_shape':list(clean.shape),'feature_mean_absolute_noise_change':float(np.mean(abs(clean-noisy))),'utterance_errors':rows,'corpus_wer':sum(r['S']+r['D']+r['I'] for r in rows)/sum(r['N'] for r in rows),'ctc_p_a':ctc_probability(np.array([[.4,.6],[.4,.6]]),[1]),'note':'No ASR system was trained; acoustic signals and error cases are controlled instructional examples.'}

if __name__=='__main__': checks(); print(experiment())
