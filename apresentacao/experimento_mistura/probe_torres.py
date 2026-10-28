"""Probe de origem: um classificador consegue dizer se um Aedes veio do Torres? (retrospectivo, didático)"""
import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.decomposition import PCA
from pathlib import Path
OUT=Path(__file__).resolve().parent
R=str(OUT.parents[1]/'results/torres_aedes_transfer_2026_10_15')+'/'
z=np.load(R+'features_htk.npz'); P=pd.read_csv(R+'partitions.csv')
prim=P[P.origin=='primary'].reset_index(drop=True); assert len(prim)==1494
Xp=z['primary']; Xt=z['torres']
def probe(X,y,g,seed=7):
    pr=np.zeros(len(y)); sg=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
    for tr,te in sg.split(X,y,g):
        assert not set(g[tr])&set(g[te])
        m=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=5000,class_weight='balanced')).fit(X[tr],y[tr]); pr[te]=m.predict_proba(X[te])[:,1]
    return dict(balanced_accuracy=float(balanced_accuracy_score(y,pr>.5)),auroc=float(roc_auc_score(y,pr)),n=int(len(y)),positives=int(y.sum()))
out={}
a=(prim.target==2).to_numpy()
X=np.vstack([Xp[a],Xt]); y=np.r_[np.zeros(a.sum()),np.ones(300)]; g=np.r_[prim.pcm_block[a].to_numpy(),[f't{i}' for i in range(300)]]
out['aedes_original_vs_torres']=probe(X,y,g)
X=np.vstack([Xp,Xt]); y=np.r_[np.zeros(1494),np.ones(300)]; g=np.r_[prim.pcm_block.to_numpy(),[f't{i}' for i in range(300)]]
out['tudo_vs_torres']=probe(X,y,g)
# 2D map (PCA on standardized features, descriptive only)
Z=PCA(2,random_state=0).fit_transform(StandardScaler().fit_transform(X))
lab=np.r_[np.where(prim.target==2,'aedes_original',np.where(prim.target==1,'outro_mosquito','ruido')),['torres']*300]
pd.DataFrame(dict(pc1=Z[:,0],pc2=Z[:,1],grupo=lab)).to_csv(OUT/'pca_torres.csv',index=False)
out['nota']='retrospectivo; Torres sem IDs biologicos (cada tensor = 1 grupo); atributos HTK aproximados <4 kHz de results/torres_aedes_transfer_2026_10_15'
print(json.dumps(out,indent=1)); json.dump(out,open(OUT/'probe_torres.json','w'),indent=1)
