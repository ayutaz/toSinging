import pyworld as pw
import librosa
import numpy as np
import soundfile as sf
import vuv_model as vuv
import dtw
import pandas as pd
from noteseq import noteseq

def analyze(wavfile=None,w=None,sr=None):
    assert not (w is None) or not (wavfile is None), "Either w or wavfile should be specified"
    if w is None:
        w, sr = librosa.load(wavfile, sr=16000)
    else:
        assert not (sr is None), "When specifying w, sr must be specified"
    yf = np.array(w,dtype=np.float64)
    _f0, t = pw.harvest(yf,sr)
    f0 = pw.stonemask(yf,_f0, t, sr)
    ap = pw.d4c(yf,f0,t,sr)
    sp = pw.cheaptrick(yf,f0,t,sr)
    return({"spectrum": sp, "aperiodicity": ap, "f0": f0, "rate": sr})

def synthesize(anl):
    return pw.synthesize(anl["f0"],anl["spectrum"],anl["aperiodicity"],anl["rate"])

INPUT = "trump.wav"
w, sr = librosa.load(INPUT,sr=16000)
anl = analyze(w=w,sr=sr)

# src_notes = pd.DataFrame({
#         'pitch': ["C3", "D3", "E3", "r", "C3", "D3", "E3", "r", 
#                   "G3", "E3", "D3", "C3", "D3", "E3", "D3", "r"],
#         'duration': [12]*16
#     })
src_notes = pd.read_csv("trump.csv")
notes,pitches = noteseq(src_notes)
mapper = {0:0, 1:10, 2:2}
model = vuv.VUVmodel("hubert_sad_20ms_model.pth","facebook/hubert-base-ls960",3,16000) 
vuv0 = model.predict(INPUT)
vuv1 = [mapper[x] for x in vuv0]
opt = dtw.dtw(vuv1,notes,uv_val=2)
N = len(pitches)
sp = np.zeros((N*4,anl["spectrum"].shape[1]))
ap = np.zeros((N*4,anl["aperiodicity"].shape[1]))
f0 = np.zeros(N*4)
for i in range(N):
    for j in range(4):
        sp[i*4+j,:] = anl["spectrum"][opt[i][0]*4+j,:]
        ap[i*4+j,:] = anl["aperiodicity"][opt[i][0]*4+j,:]
        #f0[i*4+j] = anl["f0"][opt[i][0]*4+j]
        f0[i*4+j] = pitches[opt[i][1]]
anl2 = {"f0":f0,"spectrum":sp, "aperiodicity":ap, "rate":sr}
w2 = synthesize(anl2)
sf.write("trump_out.wav",w2,anl["rate"])

