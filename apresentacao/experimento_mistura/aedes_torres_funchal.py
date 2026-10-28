"""Aedes = só Torres + Funchal. Outras fontes entram só com mosquitos NÃO-Aedes (e, numa variante, ruído).
Funchal agrupado por aparelho/taxa (Aedes e Culex do mesmo aparelho caem juntos no teste)."""
from pathlib import Path
import json, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
OUT = Path(__file__).resolve().parent
z = np.load(OUT / 'pool.npz', allow_pickle=True); X0, y, src, grp = z['X'], z['y'], z['src'], z['grp'].astype(object)
# Funchal: aparelho = taxa de amostragem da pasta (8k / 44.1k / 48k)
f = src == 'funchal'
grp[f] = ['funchal:' + ('8k' if '8khz' in g else '44k' if '44.1khz' in g else '48k') for g in grp[f]]
keep_aedes = (y == 2) & np.isin(src, ['torres', 'funchal'])
other_mosq = y == 1
noise = y == 0
def norm(X, m):
    Xn = X.copy()
    for s in np.unique(src[m]): k = m & (src == s); Xn[k] = (X[k] - X[k].mean(0)) / (X[k].std(0) + 1e-6)
    return Xn
def weights(idx, yy):
    ss = src[idx]; w = np.ones(len(idx))
    for c in np.unique(yy):
        mc = yy == c; S = np.unique(ss[mc])
        for s in S: k = mc & (ss == s); w[k] = 1 / k.sum() / len(S)
    return w / w.mean()
hgb = lambda s: HistGradientBoostingClassifier(learning_rate=.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0, min_samples_leaf=12, random_state=s)
et = lambda s: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=8, random_state=s)
def proba(kind, X, tr, te, s, ternary):
    if kind == 'dois_estagios' and ternary:
        y1 = (y[tr] > 0).astype(int); p_m = hgb(s).fit(X[tr], y1, sample_weight=weights(tr, y1)).predict_proba(X[te])[:, 1]
        mos = tr[y[tr] > 0]; y2 = (y[mos] == 2).astype(int)
        p_a = (hgb(s).fit(X[mos], y2, sample_weight=weights(mos, y2)).predict_proba(X[te])[:, 1] + et(s).fit(X[mos], y2, sample_weight=weights(mos, y2)).predict_proba(X[te])[:, 1]) / 2
        return np.c_[1 - p_m, p_m * (1 - p_a), p_m * p_a][:, 2]
    yy = y[tr] if ternary else (y[tr] == 2).astype(int)
    p = hgb(s).fit(X[tr], yy, sample_weight=weights(tr, yy)).predict_proba(X[te])
    if kind == 'dois_estagios':  # binário: média com Extra Trees
        p = (p + et(s).fit(X[tr], yy, sample_weight=weights(tr, yy)).predict_proba(X[te])) / 2
    return p[:, -1]
res = {}
for task, mask in [('ternario (com ruído)', keep_aedes | other_mosq | noise), ('binario (Aedes x outros mosquitos)', keep_aedes | other_mosq)]:
    ternary = task.startswith('ternario')
    for label, normed, kind in [('sem normalização', False, 'hgb'), ('normalização por fonte', True, 'hgb'), ('normalização + dois estágios/ensemble', True, 'dois_estagios')]:
        X = norm(X0, mask) if normed else X0
        idx = np.flatnonzero(mask); runs = []
        for s in [11, 22, 33]:
            sc = np.zeros(len(y))
            for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X[idx], y[idx], grp[idx]):
                tr, te = idx[a], idx[b]; assert not set(grp[tr]) & set(grp[te]); sc[te] = proba(kind, X, tr, te, s, ternary)
            i = idx; ae = y[i] == 2; call = sc[i] > .5
            fun = src[i] == 'funchal'
            runs.append(dict(auroc=roc_auc_score(ae, sc[i]), auroc_funchal_aedes_x_culex=roc_auc_score(ae[fun], sc[i][fun]),
                             aedes_torres=(call[ae & (src[i] == 'torres')]).mean(), aedes_funchal=(call[ae & fun]).mean(),
                             precisao=ae[call].mean(), outros_chamados_aedes=call[y[i] == 1].mean(),
                             culex_funchal_chamados_aedes=call[(y[i] == 1) & fun].mean(),
                             ruido_chamado_aedes=call[y[i] == 0].mean() if ternary else float('nan'),
                             acc_balanceada_aedes=balanced_accuracy_score(ae, call)))
        out = {k: round(float(np.mean([r[k] for r in runs])), 3) for k in runs[0]}
        out['n'] = dict(aedes_torres=300, aedes_funchal=int((keep_aedes & f).sum()), outros=int(other_mosq.sum()), ruido=int(noise.sum()) if ternary else 0)
        res[f'{task} | {label}'] = out; print(f'{task} | {label}:', json.dumps(out, ensure_ascii=False), flush=True)
(OUT / 'aedes_torres_funchal.json').write_text(json.dumps(res, indent=1, ensure_ascii=False))
