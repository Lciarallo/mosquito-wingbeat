"""Predições fora da amostra do modelo final -> matriz de confusão 3x3, curva ROC, precisão/recall por limiar."""
from pathlib import Path
import json, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve
OUT = Path(__file__).resolve().parent
z = np.load(OUT / 'pool.npz', allow_pickle=True); X0, y, src, grp = z['X'], z['y'], z['src'], z['grp'].astype(object)
f = src == 'funchal'; grp[f] = ['funchal:' + ('8k' if '8khz' in g else '44k' if '44.1khz' in g else '48k') for g in grp[f]]
mask = ((y == 2) & np.isin(src, ['torres', 'funchal'])) | (y < 2)
def norm(X):
    Xn = X.copy()
    for s in np.unique(src[mask]): k = mask & (src == s); Xn[k] = (X[k] - X[k].mean(0)) / (X[k].std(0) + 1e-6)
    return Xn
def weights(idx, yy):
    ss = src[idx]; w = np.ones(len(idx))
    for c in np.unique(yy):
        mc = yy == c; S = np.unique(ss[mc])
        for s in S: k = mc & (ss == s); w[k] = 1 / k.sum() / len(S)
    return w / w.mean()
hgb = lambda s: HistGradientBoostingClassifier(learning_rate=.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0, min_samples_leaf=12, random_state=s)
et = lambda s: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=8, random_state=s)
X = norm(X0); idx = np.flatnonzero(mask); P = np.zeros((len(y), 3))
for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=11).split(X[idx], y[idx], grp[idx]):
    tr, te = idx[a], idx[b]
    y1 = (y[tr] > 0).astype(int); p_m = hgb(11).fit(X[tr], y1, sample_weight=weights(tr, y1)).predict_proba(X[te])[:, 1]
    mos = tr[y[tr] > 0]; y2 = (y[mos] == 2).astype(int)
    p_a = (hgb(11).fit(X[mos], y2, sample_weight=weights(mos, y2)).predict_proba(X[te])[:, 1] + et(11).fit(X[mos], y2, sample_weight=weights(mos, y2)).predict_proba(X[te])[:, 1]) / 2
    P[te] = np.c_[1 - p_m, p_m * (1 - p_a), p_m * p_a]
i = idx; yt = y[i]; pa = P[i, 2]; ae = yt == 2
pred = np.where(pa > .5, 2, P[i, :2].argmax(1))
cm = confusion_matrix(yt, pred, labels=[0, 1, 2])
fpr, tpr, thr = roc_curve(ae, pa); keep = np.unique(np.linspace(0, len(fpr) - 1, 120).round().astype(int))
prec, rec, thr2 = precision_recall_curve(ae, pa)
ts = np.round(np.arange(.05, .96, .05), 2)
por_limiar = [dict(limiar=float(t), precisao=float(ae[pa > t].mean()) if (pa > t).any() else None, recall=float((pa[ae] > t).mean()),
                   fp_outros=float((pa[yt == 1] > t).mean()), fp_ruido=float((pa[yt == 0] > t).mean())) for t in ts]
hist_bins = np.linspace(0, 1, 21)
out = dict(classes=['ruido', 'outro_mosquito', 'aedes'], matriz_confusao=cm.tolist(), auroc=float(roc_auc_score(ae, pa)),
           roc=dict(fpr=fpr[keep].round(4).tolist(), tpr=tpr[keep].round(4).tolist()),
           por_limiar=por_limiar,
           histograma=dict(bins=hist_bins.round(2).tolist(), aedes=np.histogram(pa[ae], hist_bins)[0].tolist(), outros=np.histogram(pa[yt == 1], hist_bins)[0].tolist(), ruido=np.histogram(pa[yt == 0], hist_bins)[0].tolist()))
(OUT / 'curvas_modelo_final.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps({k: out[k] for k in ['matriz_confusao', 'auroc']}, indent=1)); print([ (d['limiar'], round(d['precisao'] or 0,3), round(d['recall'],3), round(d['fp_outros'],3)) for d in por_limiar])
