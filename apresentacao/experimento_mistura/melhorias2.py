"""Rodada 2: ensemble e dois estágios, com normalização por fonte; 3 seeds; AUROC dentro de cada fonte."""
from pathlib import Path
import json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
OUT = Path(__file__).resolve().parent
z = np.load(OUT / 'pool.npz', allow_pickle=True); X0, y, src, grp = z['X'], z['y'], z['src'], z['grp']
ORIG = np.isin(src, ['dryad', 'humbug', 'esc50'])
def norm(X):
    Xn = X.copy()
    for s in np.unique(src): m = src == s; Xn[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-6)
    return Xn
def weights(idx, yy=None):
    yy = y[idx] if yy is None else yy; ss = src[idx]; w = np.ones(len(idx))
    for c in np.unique(yy):
        mc = yy == c; S = np.unique(ss[mc])
        for s in S: m = mc & (ss == s); w[m] = 1.0 / m.sum() / len(S)
    return w / w.mean()
hgb = lambda s: HistGradientBoostingClassifier(learning_rate=.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0, min_samples_leaf=12, random_state=s)
et = lambda s: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=8, random_state=s)
def proba(kind, X, tr, te, s):
    if kind == 'ens':
        return (hgb(s).fit(X[tr], y[tr], sample_weight=weights(tr)).predict_proba(X[te]) +
                et(s).fit(X[tr], y[tr], sample_weight=weights(tr)).predict_proba(X[te])) / 2
    if kind == 'dois_estagios':  # 1) tem mosquito?  2) entre mosquitos, é Aedes?
        y1 = (y[tr] > 0).astype(int); m1 = hgb(s).fit(X[tr], y1, sample_weight=weights(tr, y1)); p_mosq = m1.predict_proba(X[te])[:, 1]
        mos = tr[y[tr] > 0]; y2 = (y[mos] == 2).astype(int)
        p_ae = (hgb(s).fit(X[mos], y2, sample_weight=weights(mos, y2)).predict_proba(X[te])[:, 1] +
                et(s).fit(X[mos], y2, sample_weight=weights(mos, y2)).predict_proba(X[te])[:, 1]) / 2
        return np.c_[1 - p_mosq, p_mosq * (1 - p_ae), p_mosq * p_ae]
    return hgb(s).fit(X[tr], y[tr], sample_weight=weights(tr)).predict_proba(X[te])
def oof(kind, mask, X, s):
    idx = np.flatnonzero(mask); P = np.zeros((len(y), 3))
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X[idx], y[idx], grp[idx]):
        tr, te = idx[a], idx[b]; assert not set(grp[tr]) & set(grp[te]); P[te] = proba(kind, X, tr, te, s)
    return P
def thr_from_inner(kind, mask, X, s):  # limiar escolhido em validação interna, não no teste
    return None
def summarize(P, mask, t):
    i = np.flatnonzero(mask); yt = y[i]; ae = yt == 2
    pred = np.where(P[i, 2] > t, 2, P[i, :2].argmax(1))
    within = {}
    for s in np.unique(src[i]):
        m = src[i] == s
        if ae[m].any() and (~ae[m]).any(): within[s] = round(float(roc_auc_score(ae[m], P[i][m, 2])), 3)
    return dict(auroc=float(roc_auc_score(ae, P[i, 2])), auroc_dentro_da_fonte=within,
                auroc_dentro_media=float(np.mean(list(within.values()))),
                bal_acc=float(balanced_accuracy_score(yt, pred)), aedes_recall=float((pred[ae] == 2).mean()),
                aedes_precision=float((yt[pred == 2] == 2).mean()), fp_outros=float((pred[yt == 1] == 2).mean()), fp_ruido=float((pred[yt == 0] == 2).mean()),
                recall_originais=f"{int((pred[ae & np.isin(src[i], ['dryad','humbug'])] == 2).sum())}/{int((ae & np.isin(src[i], ['dryad','humbug'])).sum())}")
res = {}
Xn = norm(X0)
for label, kind, mask, X in [('antes: só fontes originais, sem normalização', 'hgb', ORIG, X0),
                             ('todas as fontes, sem normalização', 'hgb', np.ones(len(y), bool), X0),
                             ('todas + normalização por fonte', 'hgb', np.ones(len(y), bool), Xn),
                             ('todas + normalização + ensemble', 'ens', np.ones(len(y), bool), Xn),
                             ('todas + normalização + dois estágios', 'dois_estagios', np.ones(len(y), bool), Xn)]:
    runs = []
    for s in [11, 22, 33]:
        P = oof(kind, mask, X, s)
        # limiar: o que maximiza acurácia balanceada Aedes×resto nas OUTRAS seeds não existe ainda; usa regra fixa 1/3 (pré-definida)
        runs.append(summarize(P, mask, 1 / 3))
    keys = [k for k in runs[0] if isinstance(runs[0][k], float)]
    res[label] = {k: [round(float(np.mean([r[k] for r in runs])), 3), round(float(np.std([r[k] for r in runs])), 3)] for k in keys}
    res[label]['auroc_dentro_da_fonte(seed11)'] = runs[0]['auroc_dentro_da_fonte']; res[label]['recall_originais(seeds)'] = [r['recall_originais'] for r in runs]
    print(label, json.dumps(res[label], ensure_ascii=False), flush=True)
(OUT / 'melhorias2.json').write_text(json.dumps(res, indent=1, ensure_ascii=False))
