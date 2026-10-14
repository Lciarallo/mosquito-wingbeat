"""Explicit support-only and strict-union F1; recording-level paired intervals."""
import numpy as np
from sklearn.metrics import f1_score


def classification_metrics(y, pred, accepted=None):
    y, pred = np.asarray(y).astype(str), np.asarray(pred).astype(str)
    if len(y) == 0 or len(y) != len(pred):
        raise ValueError('Empty or misaligned labels.')
    support = np.unique(y)
    union = np.union1d(support, np.unique(pred))
    correct = y == pred
    accepted = np.ones(len(y),bool) if accepted is None else np.asarray(accepted,bool)
    if len(accepted) != len(y):
        raise ValueError('Misaligned acceptance.')
    a = float(f1_score(y,pred,labels=support,average='macro',zero_division=0))
    b = float(f1_score(y,pred,labels=union,average='macro',zero_division=0))
    return dict(n=len(y), correct=int(correct.sum()), accuracy=float(correct.mean()),
                balanced_accuracy=float(np.mean([correct[y==k].mean() for k in support])),
                macro_f1_support_only=a, macro_f1_strict_union=b,
                macro_f1_support_minus_strict=a-b,
                predicted_taxa_outside_support=sorted(set(pred)-set(y)),
                coverage=float(accepted.mean()), emitted=int(accepted.sum()),
                accepted_accuracy=float(correct[accepted].mean()) if accepted.any() else None,
                correct_emissions_over_all=float((accepted & correct).mean()))


def bootstrap_metrics(y, predictions, acceptances, repetitions=2000, seed=20261013, clusters=None):
    """Same stratified recording draws for all models, preserving observed taxa.

    Undefined accepted accuracy is NaN, never zero. One recording has no CI.
    Each input row must already be exactly one recording/collision group.
    """
    y = np.asarray(y).astype(str)
    keys = ['accuracy','balanced_accuracy','macro_f1_strict_union','coverage','accepted_accuracy']
    if len(y)<2:
        return {'intervals': {}, 'paired_deltas': {}, 'reason': 'One recording; no independent recording bootstrap.'}
    rng = np.random.default_rng(seed)
    blocks=np.arange(len(y)).astype(str) if clusters is None else np.asarray(clusters).astype(str)
    block_ids=np.unique(blocks)
    if len(block_ids)<2:
        return {'intervals':{},'paired_deltas':{},'reason':'One independent PCM collision block; no recording CI.'}
    block_rows=[np.flatnonzero(blocks==g) for g in block_ids]
    signatures=np.array(['|'.join(np.unique(y[ix])) for ix in block_rows])
    strata=[np.flatnonzero(signatures==label) for label in np.unique(signatures)]
    names = list(predictions)
    # Vectorised confusion matrices make 20-class paired bootstrap inexpensive.
    labels = np.unique(np.concatenate([y]+[np.asarray(predictions[n]).astype(str) for n in names]))
    yi = np.searchsorted(labels,y)
    support = np.unique(yi)
    arrays = {n:np.full((repetitions,len(keys)),np.nan) for n in names}
    for b in range(repetitions):
        sampled=np.concatenate([rng.choice(s,len(s),replace=True) for s in strata])
        ix=np.concatenate([block_rows[g] for g in sampled])
        truth = yi[ix]
        for n in names:
            pi = np.searchsorted(labels,np.asarray(predictions[n]).astype(str)[ix])
            a = np.asarray(acceptances[n],bool)[ix]
            cm = np.bincount(truth*len(labels)+pi,minlength=len(labels)**2).reshape(len(labels),len(labels))
            tp=np.diag(cm); rows=cm.sum(1); cols=cm.sum(0)
            union=(rows+cols)>0
            f1=np.divide(2*tp,rows+cols,out=np.zeros(len(labels)),where=union)
            correct=truth==pi
            arrays[n][b]=[correct.mean(),np.mean(tp[support]/rows[support]),
                          f1[union].mean(),a.mean(),correct[a].mean() if a.any() else np.nan]
    def interval(v):
        valid=v[np.isfinite(v)]
        return dict(low=float(np.quantile(valid,.025)) if len(valid) else None,
                    high=float(np.quantile(valid,.975)) if len(valid) else None,
                    valid_replicates=len(valid), total_replicates=repetitions)
    intervals={n:{k:interval(arrays[n][:,j]) for j,k in enumerate(keys)} for n in names}
    base=names[0]
    delta={n:{k:interval(arrays[n][:,j]-arrays[base][:,j]) for j,k in enumerate(keys)} for n in names[1:]}
    return dict(intervals=intervals,paired_deltas=delta,reference_model=base,
                seed=seed,repetitions=repetitions,unit='recording/collision group',
                collision_blocks=len(block_ids),
                scheme='paired collision-block percentile bootstrap, stratified by observed label-set of each block; union recomputed per replicate')
