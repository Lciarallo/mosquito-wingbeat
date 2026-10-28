"""Interpretabilidade do modelo final (Aedes só de Torres + Funchal; normalização por fonte;
pesos por fonte; dois estágios). Importância por banda de frequência via permutação no teste."""
from pathlib import Path
import json, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
OUT = Path(__file__).resolve().parent
z = np.load(OUT / 'pool.npz', allow_pickle=True); X0, y, src, grp = z['X'], z['y'], z['src'], z['grp'].astype(object)
f = src == 'funchal'; grp[f] = ['funchal:' + ('8k' if '8khz' in g else '44k' if '44.1khz' in g else '48k') for g in grp[f]]
mask = ((y == 2) & np.isin(src, ['torres', 'funchal'])) | (y < 2)
EDGES = 700 * (10 ** (np.linspace(0, 2595 * np.log10(1 + 24000 / 700), 34) / 2595) - 1)
CENTERS = EDGES[1:17]                     # 16 bandas mantidas (até 4 kHz)
def cols(b): return [b + 16 * k for k in range(5)]   # média, desvio, p10, p50, p90 da banda b
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
X = norm(X0); idx = np.flatnonzero(mask); rng = np.random.default_rng(0)
base_sc = np.zeros(len(y)); perm_sc = np.zeros((16, len(y)))
for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=11).split(X[idx], y[idx], grp[idx]):
    tr, te = idx[a], idx[b]
    y1 = (y[tr] > 0).astype(int); m1 = hgb(11).fit(X[tr], y1, sample_weight=weights(tr, y1))
    mos = tr[y[tr] > 0]; y2 = (y[mos] == 2).astype(int)
    m2a = hgb(11).fit(X[mos], y2, sample_weight=weights(mos, y2)); m2b = et(11).fit(X[mos], y2, sample_weight=weights(mos, y2))
    score = lambda Z: m1.predict_proba(Z)[:, 1] * (m2a.predict_proba(Z)[:, 1] + m2b.predict_proba(Z)[:, 1]) / 2
    base_sc[te] = score(X[te])
    for band in range(16):
        Z = X[te].copy(); c = cols(band); Z[:, c] = Z[rng.permutation(len(te))][:, c]; perm_sc[band, te] = score(Z)
i = idx; ae = y[i] == 2; fun = src[i] == 'funchal'
base = roc_auc_score(ae, base_sc[i]); base_f = roc_auc_score(ae[fun], base_sc[i][fun])
bands = [dict(centro_hz=round(float(CENTERS[b])), faixa_hz=[round(float(EDGES[b])), round(float(EDGES[b + 2]))],
              queda_auroc=round(float(base - roc_auc_score(ae, perm_sc[b, i])), 4),
              queda_auroc_aedes_x_culex=round(float(base_f - roc_auc_score(ae[fun], perm_sc[b, i][fun])), 4)) for b in range(16)]
# a origem ainda é decodificável depois da normalização? (só outros mosquitos, chance = 1/nº de fontes)
om = mask & (y == 1); S = src[om]; Xo = X[om]; go = grp[om]; pr = np.empty(len(S), object)
for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=7).split(Xo, S, go):
    pr[b] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight='balanced')).fit(Xo[a], S[a]).predict(Xo[b])
out = dict(auroc_base=round(float(base), 3), auroc_base_aedes_x_culex=round(float(base_f), 3), bandas=bands,
           probe_origem_outros_mosquitos=dict(fontes=sorted(set(S)), acc_balanceada=round(float(balanced_accuracy_score(S, pr)), 3), acaso=round(1 / len(set(S)), 3)))
(OUT / 'interpretabilidade_final.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps(out, indent=1, ensure_ascii=False))
