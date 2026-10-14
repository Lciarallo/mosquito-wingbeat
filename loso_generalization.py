"""Leave-one-source-out (LOSO) benchmark for the Aedes aegypti vs not-Aedes detector.

Candidates are declared in CANDIDATES before any run. The held-out source is never used for
fitting, feature selection or threshold choice. Per-source normalisation variants use only
UNLABELLED statistics of the held-out source (what a device could collect in the field).
Windows are not independent: AUC confidence intervals resample whole recording groups.
"""
import hashlib,json,sys,time
from datetime import datetime,timezone
import numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from train_reproduction import ROOT
from aedes_binary import targets,weights

SEGMENTS=ROOT/'results/sex_balanced_reliability_2026_10_01/segments.csv'
FEATURES=ROOT/'data/cache/reliability_2026_10_01/features.npy'
OUT=ROOT/'results/loso_corrected_2026_10_07'
HELD_OUT=['dryad','humbug','mosquitosong']  # sources with both classes
ACOUSTIC={'dryad','humbug','mosquitosong','figshare','esc50'}
# Dryad 10.5061/dryad.98d7s contains mobile-phone acoustic recordings.
# The optical corpus in Fanioudakis et al. is a different dataset.
SEED=20261003


def sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def load():
    f=pd.read_csv(SEGMENTS,low_memory=False);x=np.load(FEATURES).astype('float64')
    if len(f)!=len(x):raise ValueError('Feature rows and segment manifest differ.')
    keep=f.quality_ok.to_numpy()&(targets(f)>=0)&f.domain.ne('ufrgs').to_numpy()
    f=f[keep].reset_index(drop=True);x=x[keep]
    if not np.isfinite(x).all():raise ValueError('Non-finite features: repair the extraction before evaluation.')
    for column in ['group','content_sha256']:
        if f.dropna(subset=[column]).groupby(column).domain.nunique().gt(1).any():
            raise ValueError('Cross-source overlap in '+column)
    return f,x,targets(f)


def source_standardise(x,domains):
    out=np.empty_like(x)
    for d in np.unique(domains):
        m=domains==d;mu=x[m].mean(0);sd=x[m].std(0)
        out[m]=(x[m]-mu)/np.where(sd<1e-8,1,sd)
    return np.clip(out,-8,8)


def consistent_features(x,y,f,k):
    """Keep features whose class direction agrees across every training source (training only)."""
    y=np.asarray(y)
    scores=[];signs=[]
    for d in f.domain.unique():
        m=(f.domain==d).to_numpy()
        if len(np.unique(y[m]))<2:continue
        r=pd.DataFrame(x[m]).rank().to_numpy();pos=y[m]==1
        n1,n0=pos.sum(),(~pos).sum()
        auc=(r[pos].sum(0)-n1*(n1+1)/2)/(n1*n0)
        signs.append(np.sign(auc-.5));scores.append(np.abs(auc-.5))
    if len(signs)<2:raise ValueError('Feature consistency requires two training sources with both classes.')
    signs=np.array(signs);scores=np.array(scores)
    agree=(signs==signs[0]).all(0)
    worst=np.where(agree,scores.min(0),-1)
    eligible=np.flatnonzero(agree & (scores.min(0)>0))
    if not len(eligible):raise ValueError('No features agree across training sources.')
    return eligible[np.argsort(-worst[eligible],kind='stable')[:k]]


def make(name):
    if name.startswith('et'):return ExtraTreesClassifier(n_estimators=300,max_depth=16,min_samples_leaf=3,max_features=.5,n_jobs=24,random_state=SEED)
    if name.startswith('hgb'):return HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=15,l2_regularization=3,learning_rate=.05,early_stopping=False,random_state=SEED)
    return LogisticRegression(C=.05,max_iter=3000)


# name -> (model family, options)
CANDIDATES={
    'et_baseline':dict(),
    'hgb_baseline':dict(),
    'lr_baseline':dict(scale=True),
    'et_no_positive_only_source':dict(drop=['figshare']),
    'hgb_acoustic_only':dict(drop=['figshare'],acoustic_only=True),
    'et_consistent40':dict(consistent=40,drop=['figshare']),
    'lr_consistent40':dict(consistent=40,drop=['figshare'],scale=True),
    'et_source_norm':dict(srcnorm=True,drop=['figshare']),
    'lr_source_norm':dict(srcnorm=True,drop=['figshare'],scale=True),
    'hgb_source_norm':dict(srcnorm=True,drop=['figshare']),
}


def run(name,opt,f,x,y,held):
    test=(f.domain==held).to_numpy();train=~test
    if opt.get('drop'):train&=~f.domain.isin(opt['drop']).to_numpy()
    if opt.get('acoustic_only'):train&=f.domain.isin(ACOUSTIC).to_numpy()
    xx=source_standardise(x,f.domain.to_numpy()) if opt.get('srcnorm') else x
    cols=np.arange(x.shape[1])
    if opt.get('consistent'):cols=consistent_features(xx[train],y[train],f[train].reset_index(drop=True),opt['consistent'])
    xt=xx[train][:,cols];xe=xx[test][:,cols]
    if opt.get('scale'):
        s=StandardScaler().fit(xt);xt=np.clip(s.transform(xt),-8,8);xe=np.clip(s.transform(xe),-8,8)
    w=weights(f[train].reset_index(drop=True))
    m=make(name).fit(xt,y[train],sample_weight=w)
    return m.predict_proba(xe)[:,1],test


def group_auc_ci(y,p,groups,rng,n=500):
    ug=np.unique(groups);idx={g:np.flatnonzero(groups==g) for g in ug};out=[]
    for _ in range(n):
        sel=np.concatenate([idx[g] for g in rng.choice(ug,len(ug))])
        if len(np.unique(y[sel]))==2:out.append(roc_auc_score(y[sel],p[sel]))
    if not out:return None,None
    return float(np.percentile(out,2.5)),float(np.percentile(out,97.5))


def main():
    only=sys.argv[1:] or list(CANDIDATES)
    if any(name not in CANDIDATES for name in only):raise ValueError('Unknown candidate.')
    if OUT.exists():raise FileExistsError('Use a new output directory; never overwrite evaluated results.')
    f,x,y=load();OUT.mkdir();rng=np.random.default_rng(SEED);rows=[];preds={}
    (OUT/'protocol.json').write_text(json.dumps(dict(candidates=CANDIDATES,held_out=HELD_OUT,seed=SEED,
        created_utc=datetime.now(timezone.utc).isoformat(),
        inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in [SEGMENTS,FEATURES]},
        code_sha256=sha(ROOT/'loso_generalization.py'),
        executed_candidates=only,acoustic_sources=sorted(ACOUSTIC),
        source_normalisation='Transductive: uses unlabelled statistics of the entire evaluated source, including test windows.',
        note='Retrospective corrected analysis on previously examined sources; no independent confirmation or device approval.'),indent=2))
    for name in only:
        for held in HELD_OUT:
            t=time.time();p,test=run(name,CANDIDATES[name],f,x,y,held);yt=y[test];g=f.group[test].to_numpy()
            auc=roc_auc_score(yt,p);lo,hi=group_auc_ci(yt,p,g,rng)
            # threshold-free score plus balanced accuracy at the 0.5 cut of source-balanced training
            pred=p>=.5;bal=.5*(pred[yt==1].mean()+(~pred[yt==0]).mean())
            rows.append(dict(model=name,held_out=held,auc=auc,auc_lo95=lo,auc_hi95=hi,balanced_accuracy_at_half=bal,recall=pred[yt==1].mean(),specificity=(~pred[yt==0]).mean(),positives=int((yt==1).sum()),negatives=int((yt==0).sum()),groups=int(len(np.unique(g)))))
            preds[f'{name}|{held}']=p
            print(f'{name:28s} {held:13s} AUC {auc:.3f} [{lo:.3f},{hi:.3f}] bal@.5 {bal:.3f} ({time.time()-t:.0f}s)',flush=True)
    pd.DataFrame(rows).to_csv(OUT/'loso_metrics.csv',index=False)
    np.savez_compressed(OUT/'loso_predictions.npz',**preds)
    s=pd.DataFrame(rows).pivot(index='model',columns='held_out',values='auc');s['mean']=s.mean(axis=1);s['worst']=s.min(axis=1)
    print(s.sort_values('worst',ascending=False).round(3).to_string())


if __name__=='__main__':main()
