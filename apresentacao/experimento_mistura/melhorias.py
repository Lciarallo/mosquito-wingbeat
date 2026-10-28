"""Tentativas de melhoria do detector ternário (ruído / outro mosquito / Aedes) com todas as fontes."""
from pathlib import Path
import json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
OUT = Path(__file__).resolve().parent; SEED = 20261027
z = np.load(OUT / 'pool.npz', allow_pickle=True); X0, y, src, grp = z['X'], z['y'], z['src'], z['grp']
ORIG = np.isin(src, ['dryad', 'humbug', 'esc50'])

def per_source_norm(X):  # normalização por fonte/aparelho (não usa rótulos)
    Xn = X.copy()
    for s in np.unique(src):
        m = src == s; mu, sd = X[m].mean(0), X[m].std(0) + 1e-6; Xn[m] = (X[m] - mu) / sd
    return Xn

def weights(idx, balance_source):
    w = np.ones(len(idx)); yy, ss = y[idx], src[idx]
    for c in np.unique(yy):
        mc = yy == c
        if balance_source:
            for s in np.unique(ss[mc]): m = mc & (ss == s); w[m] = 1.0 / m.sum() / len(np.unique(ss[mc]))
        else: w[mc] = 1.0 / mc.sum()
    return w / w.mean()

MODELS = {
    'histgb': lambda: HistGradientBoostingClassifier(learning_rate=.06, max_iter=250, max_leaf_nodes=15, l2_regularization=1.0, min_samples_leaf=12, random_state=SEED),
    'logistica': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=.5, max_iter=5000)),
    'extra_trees': lambda: ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=8, random_state=SEED),
    'mlp': lambda: make_pipeline(StandardScaler(), MLPClassifier((128, 64), alpha=1e-3, max_iter=600, early_stopping=True, random_state=SEED)),
}
def fit(name, X, idx, bal):
    m = MODELS[name]()
    w = weights(idx, bal)
    if name == 'mlp': return m.fit(X[idx], y[idx])  # MLP do sklearn não aceita peso: reamostra
    if name == 'logistica': return m.fit(X[idx], y[idx], logisticregression__sample_weight=w)
    return m.fit(X[idx], y[idx], sample_weight=w)

def choose_threshold(name, X, idx, bal):
    """Limiar de Aedes escolhido só no treino (validação interna por grupo): maximiza acurácia balanceada."""
    p = np.zeros(len(idx)); inner = StratifiedGroupKFold(3, shuffle=True, random_state=SEED + 1)
    for a, b in inner.split(X[idx], y[idx], grp[idx]):
        p[b] = fit(name, X, idx[a], bal).predict_proba(X[idx[b]])[:, 2]
    best = max(np.linspace(.05, .9, 35), key=lambda t: balanced_accuracy_score(y[idx] == 2, p > t))
    return float(best)

def evaluate(name, mask, norm=False, bal=False, thr=False, seed=SEED):
    X = per_source_norm(X0) if norm else X0
    idx_all = np.flatnonzero(mask); P = np.zeros((len(y), 3)); pred = np.full(len(y), -1)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X[idx_all], y[idx_all], grp[idx_all]):
        tr, te = idx_all[tr], idx_all[te]; assert not set(grp[tr]) & set(grp[te])
        m = fit(name, X, tr, bal); P[te] = m.predict_proba(X[te])
        if thr:
            t = choose_threshold(name, X, tr, bal)
            pred[te] = np.where(P[te, 2] > t, 2, P[te, :2].argmax(1))
        else: pred[te] = P[te].argmax(1)
    i = idx_all; yt, pt = y[i], pred[i]; ae = yt == 2
    r = dict(auroc_aedes=float(roc_auc_score(ae, P[i, 2])), bal_acc_3classes=float(balanced_accuracy_score(yt, pt)),
             acc=float((yt == pt).mean()), aedes_recall=float((pt[ae] == 2).mean()),
             aedes_precision=float((yt[pt == 2] == 2).mean()) if (pt == 2).any() else None,
             fp_outros=float((pt[yt == 1] == 2).mean()), fp_ruido=float((pt[yt == 0] == 2).mean()),
             por_fonte={s: f'{int((pt[ae & (src[i] == s)] == 2).sum())}/{int((ae & (src[i] == s)).sum())}' for s in np.unique(src[i][ae])})
    return r

res = {}
res['V0_antes_original_histgb'] = evaluate('histgb', ORIG)
for name in MODELS:
    res[f'V1_todas_{name}'] = evaluate(name, np.ones(len(y), bool))
for name in ['histgb', 'extra_trees', 'logistica']:
    res[f'V2_todas_norm_{name}'] = evaluate(name, np.ones(len(y), bool), norm=True)
    res[f'V3_todas_norm_bal_thr_{name}'] = evaluate(name, np.ones(len(y), bool), norm=True, bal=True, thr=True)
res['V3_original_norm_bal_thr_histgb'] = evaluate('histgb', ORIG, norm=True, bal=True, thr=True)
(OUT / 'melhorias.json').write_text(json.dumps(res, indent=1, ensure_ascii=False))
rows = [dict(variante=k, **{kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in r.items() if kk != 'por_fonte'}, por_fonte=r['por_fonte']) for k, r in res.items()]
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 120)
print(pd.DataFrame(rows).to_string(index=False))
