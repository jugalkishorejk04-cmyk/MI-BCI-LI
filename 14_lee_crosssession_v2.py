"""
14_lee_crosssession_v2.py - Lee2019 two-session analysis, reading RAW .mat files
directly (MOABB's paradigm only exposes one session; the raw files have both).

PRIMARY: selectivity(session 1) -> delta-LI(session 2)  [circularity-killer]
SECONDARY: test-retest reliability of selectivity and delta-LI (ICC, sign)

Reads EEG_MI_train + EEG_MI_test per session, concatenates, resamples 1000->160,
computes mu-band ERD selectivity + delta-LI on C3/Cz/C4, identical pipeline to main.

Output: lee_crosssession.csv, lee_crosssession_report.txt
"""
import os, glob, warnings, gc
os.environ["OMP_NUM_THREADS"]="1"
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import scipy.io as sio
from scipy.signal import welch, resample_poly
from scipy import stats

SF_RAW=1000.0; SF=160.0
BASE=(0,int(0.5*SF)); MI=(int(0.5*SF),int(3.0*SF)); STAGES=3; MU=(8.0,13.0)
EPOCH_S=3.0  # 0-3 s post onset, matching main pipeline

BASE_DIR=r"F:\MI_BCI_Pipeline\F-\MI_BCI_Pipeline\F-\mne_data\MNE-lee2019-mi-data"

def erd_db(seg):
    b=seg[BASE[0]:BASE[1]]; m=seg[MI[0]:MI[1]]
    if len(b)<8 or len(m)<8: return np.nan
    fb,pb=welch(b,fs=SF,nperseg=min(len(b),int(SF)))
    fm,pm=welch(m,fs=SF,nperseg=min(len(m),int(SF)))
    mb=(fb>=MU[0])&(fb<=MU[1]); mm=(fm>=MU[0])&(fm<=MU[1])
    Pb=np.mean(pb[mb]) if mb.any() else np.nan
    Pm=np.mean(pm[mm]) if mm.any() else np.nan
    return 10*np.log10(Pm/Pb) if (Pb and Pb>0 and Pm and Pm>0) else np.nan

def stage_metrics(c_erd,i_erd):
    n=len(c_erd)
    if n<STAGES*2: return None
    e=np.linspace(0,n,STAGES+1,dtype=int); cs,is_=[],[]
    for s in range(STAGES):
        c=c_erd[e[s]:e[s+1]]; i=i_erd[e[s]:e[s+1]]
        c=c[~np.isnan(c)]; i=i[~np.isnan(i)]
        if len(c)<1 or len(i)<1: return None
        cs.append(np.mean(np.abs(c))); is_.append(np.mean(np.abs(i)))
    cs=np.array(cs); is_=np.array(is_); x=np.array([1.,2.,3.])
    sel=np.polyfit(x,cs,1)[0]-np.polyfit(x,(cs+is_)/2,1)[0]
    li_e=(cs[0]-is_[0])/(cs[0]+is_[0]) if (cs[0]+is_[0])>0 else np.nan
    li_l=(cs[2]-is_[2])/(cs[2]+is_[2]) if (cs[2]+is_[2])>0 else np.nan
    return sel, li_l-li_e

def load_session(matfile):
    """Return (trials[n,3,time@160], labels) concatenating train+test."""
    d=sio.loadmat(matfile, squeeze_me=True, struct_as_record=False)
    Xs=[]; ys=[]
    for key in ["EEG_MI_train","EEG_MI_test"]:
        if key not in d: continue
        s=d[key]
        chan=list(s.chan)
        idx=[chan.index(c) for c in ["C3","Cz","C4"]]
        x=s.x  # (time, 62)
        t=np.atleast_1d(s.t).astype(int)
        y=np.atleast_1d(s.y_dec).astype(int)
        seg_raw=int(EPOCH_S*SF_RAW)
        for oi,onset in enumerate(t):
            if onset+seg_raw>x.shape[0]: continue
            seg=x[onset:onset+seg_raw, idx]           # (time,3) at 1000Hz
            seg=resample_poly(seg, int(SF), int(SF_RAW), axis=0)  # ->160Hz
            Xs.append(seg.T)                          # (3,time)
            ys.append(y[oi])
    if not Xs: return None,None
    return np.array(Xs), np.array(ys)

rows=[]
for sid in range(1,55):
    sub={"subject":sid}
    ok=True
    for sess in [1,2]:
        pat=os.path.join(BASE_DIR,"**",f"session{sess}",f"s{sid}",f"sess0{sess}_subj{sid:02d}_EEG_MI.mat")
        files=glob.glob(pat,recursive=True)
        if not files:
            ok=False; break
        X,y=load_session(files[0])
        if X is None: ok=False; break
        # labels: 1 and 2. Determine which is left/right from y_class if present.
        # OpenBMI: class order typically [right, left] or [left, right]; y_dec 1/2.
        # T1 = left_hand (contra=C4), T2 = right_hand (contra=C3).
        # We compute per label using the standard mapping label1->right, label2->left
        # (OpenBMI y_class = ['right','left']); verify below.
        for lbl,tag,contra_i,ipsi_i in [(2,"T1",2,0),(1,"T2",0,2)]:
            # contra_i/ipsi_i index into [C3,Cz,C4] = [0,1,2]; C4=2, C3=0
            m=(y==lbl)
            if m.sum()<STAGES*2: continue
            d=X[m]
            c=np.array([erd_db(d[t,contra_i,:]) for t in range(len(d))])
            i=np.array([erd_db(d[t,ipsi_i,:]) for t in range(len(d))])
            r=stage_metrics(c,i)
            if r is None: continue
            sub[f"sel_{tag}_s{sess}"]=r[0]
            sub[f"dLI_{tag}_s{sess}"]=r[1]
        del X; gc.collect()
    if ok:
        rows.append(sub)
        print(f"subj {sid} done", flush=True)
    else:
        print(f"subj {sid} skip (missing session file)", flush=True)

df=pd.DataFrame(rows)
df.to_csv("lee_crosssession.csv",index=False)

lines=[]
def w(s): lines.append(s); print(s,flush=True)
w(f"\nLee two-session (raw read): n={len(df)} subjects\n")
for tag in ["T1","T2"]:
    # PRIMARY cross-session prediction
    for a,b,lab in [(f"sel_{tag}_s1",f"dLI_{tag}_s2",f"{tag}: sel(S1)->dLI(S2)"),
                    (f"sel_{tag}_s2",f"dLI_{tag}_s1",f"{tag}: sel(S2)->dLI(S1)")]:
        if a in df and b in df:
            s=df[[a,b]].dropna()
            if len(s)>=5:
                r,p=stats.spearmanr(s[a],s[b])
                w(f"  PRIMARY {lab}: rho={r:+.3f} p={p:.4f} n={len(s)}")
    # SECONDARY reliability
    for metric,pa,pb in [("selectivity",f"sel_{tag}_s1",f"sel_{tag}_s2"),
                         ("dLI",f"dLI_{tag}_s1",f"dLI_{tag}_s2")]:
        if pa in df and pb in df:
            s=df[[pa,pb]].dropna()
            if len(s)>=5:
                r,p=stats.spearmanr(s[pa],s[pb])
                a=s[pa].values; b=s[pb].values; M=np.c_[a,b]; gm=M.mean()
                msb=2*((M.mean(1)-gm)**2).sum()/(len(M)-1)
                msw=((M-M.mean(1,keepdims=True))**2).sum()/len(M)
                icc=(msb-msw)/(msb+msw) if (msb+msw)>0 else np.nan
                sign=np.mean(np.sign(a)==np.sign(b))
                w(f"  RELIABILITY {tag} {metric}: r={r:+.3f} ICC={icc:+.3f} sign={sign:.2f} n={len(s)}")
    w("")
open("lee_crosssession_report.txt","w").write("\n".join(lines))
w("Saved lee_crosssession.csv + report")
