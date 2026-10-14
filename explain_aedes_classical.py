"""Validation-only permutation explanations, same subset for baseline and permutations."""
import numpy as np,pandas as pd,joblib,json,hashlib
from sklearn.inspection import permutation_importance
from aedes_binary import targets,weights,ROOT
from aedes_classical import O,FEATURE_NAMES

def main():
    f=pd.read_csv(ROOT/'results/multisource/segments.csv',low_memory=False);y=targets(f)
    va=np.flatnonzero(f.split.eq('validation')&f.quality_ok&(y>=0));w=weights(f.iloc[va]);x=np.load(ROOT/'data/cache/aedes_classical_features.npy')
    rng=np.random.default_rng(20261004);subset=np.concatenate([rng.choice(np.flatnonzero(y[va]==c),min(128,int((y[va]==c).sum())),replace=False) for c in [0,1]])
    audit=json.loads((O/'training_audit.json').read_text())
    for item in audit['models']:
        name=item['model'];path=O/f'models/{name}.joblib';sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==item['sha256']
        saved=joblib.load(path);pi=permutation_importance(saved['model'],x[va[subset]],y[va[subset]],scoring='roc_auc',n_repeats=3,random_state=20261004,n_jobs=1,sample_weight=w[subset])
        pd.DataFrame(dict(feature=FEATURE_NAMES,auc_drop=pi.importances_mean,std=pi.importances_std)).sort_values('auc_drop',ascending=False).to_csv(O/f'explanation_{name}.csv',index=False)
        assert sha==hashlib.sha256(path.read_bytes()).hexdigest();print(name,'explained',flush=True)
        if name=='logistic':
            pd.DataFrame(dict(feature=FEATURE_NAMES,standardized_coefficient=saved['model'].steps[-1][1].coef_[0])).to_csv(O/'logistic_coefficients.csv',index=False)
    (O/'explanation_audit.json').write_text(json.dumps(dict(validation_only=True,rows=len(subset),repeats=3,indices=va[subset].tolist(),baseline_and_permutations_same_subset=True,model_unchanged=True,correlated_features_limit_causal_interpretation=True),indent=2))
if __name__=='__main__':main()
