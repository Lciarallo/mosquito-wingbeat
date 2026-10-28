"""Aedes só de Torres + Funchal no treino. Negativos de todas as fontes.
1) validação por grupo dentro desse conjunto; 2) Aedes das outras fontes como teste nunca visto."""
from pathlib import Path
import json, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
OUT = Path(__file__).resolve().parent
z = np.load(OUT / 'pool.npz', allow_pickle=True); X0, y, src, grp = z['X'], z['y'], z['src'], z['grp']
def norm(X):
    Xn = X.copy()
    for s in np.unique(src): m = src == s; Xn[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-6)
    return Xn
def weights(idx):
    yy, ss = y[idx], src[idx]; w = np.ones(len(idx))
    for c in np.unique(yy):
        mc = yy == c; S = np.unique(ss[mc])
        for s in S: m = mc & (ss == s); w[m] = 1 / m.sum() / len(S)
    return w / w.mean()
hgb = lambda s: HistGradientBoostingClassifier(learning_rate=.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0, min_samples_leaf=12, random_state=s)
pos_train = (y == 2) & np.isin(src, ['torres', 'funchal'])
pos_ext = (y == 2) & ~np.isin(src, ['torres', 'funchal'])
neg = y < 2
res = {}
for label, X in [('sem normalização', X0), ('com normalização por fonte', norm(X0))]:
    out = {}
    # 1) CV por grupo: Aedes Torres+Funchal x negativos de todas as fontes
    idx = np.flatnonzero(pos_train | neg); P = np.zeros((len(y), 3))
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=11).split(X[idx], y[idx], grp[idx]):
        tr, te = idx[a], idx[b]; P[te] = hgb(11).fit(X[tr], y[tr], sample_weight=weights(tr)).predict_proba(X[te])
    i = idx; ae = y[i] == 2; pred = P[i].argmax(1)
    out['cv_auroc_aedes'] = round(float(roc_auc_score(ae, P[i, 2])), 3)
    out['cv_aedes_detectados'] = {s: f'{int((pred[ae & (src[i] == s)] == 2).sum())}/{int((ae & (src[i] == s)).sum())}' for s in ['torres', 'funchal']}
    out['cv_outros_chamados_aedes'] = f'{int((pred[y[i] == 1] == 2).sum())}/{int((y[i] == 1).sum())}'
    out['cv_ruido_chamado_aedes'] = f'{int((pred[y[i] == 0] == 2).sum())}/{int((y[i] == 0).sum())}'
    fm = src[i] == 'funchal'; out['cv_auroc_dentro_do_funchal(aedes_x_culex_mesmos_aparelhos)'] = round(float(roc_auc_score(ae[fm], P[i][fm, 2])), 3)
    # 2) modelo final (Torres+Funchal Aedes + todos os negativos) aplicado aos Aedes das outras fontes
    #    negativos de teste = os mesmos OOF acima (não dá para usar negativos vistos no treino final)
    m = hgb(11).fit(X[idx], y[idx], sample_weight=weights(idx)); pe = m.predict_proba(X[pos_ext])
    out['externo_aedes_detectados'] = {s: f'{int((pe[src[pos_ext] == s].argmax(1) == 2).sum())}/{int((src[pos_ext] == s).sum())}' for s in np.unique(src[pos_ext])}
    sc_neg = P[neg, 2]; sc_pos = pe[:, 2]
    out['externo_auroc(aedes_novos_vs_negativos_OOF)'] = round(float(roc_auc_score(np.r_[np.ones(len(sc_pos)), np.zeros(len(sc_neg))], np.r_[sc_pos, sc_neg])), 3)
    res[label] = out; print(label, json.dumps(out, ensure_ascii=False, indent=1), flush=True)
(OUT / 'so_torres_funchal.json').write_text(json.dumps(res, indent=1, ensure_ascii=False))
