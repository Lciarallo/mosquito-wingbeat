"""Modelo final com SVM (RBF): mesma receita (Aedes só de Torres + Funchal; normalização e pesos por fonte; dois estágios;
validação cruzada por grupo). Gera matriz de confusão, ROC, métricas por limiar, histograma e importância por faixa."""
from pathlib import Path
import json, warnings, numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
warnings.filterwarnings('ignore')
OUT = Path(__file__).resolve().parent
z = np.load(OUT / 'pool.npz', allow_pickle=True); X0, y, src, grp = z['X'], z['y'], z['src'], z['grp'].astype(object)
f = src == 'funchal'; grp[f] = ['funchal:' + ('8k' if '8khz' in g else '44k' if '44.1khz' in g else '48k') for g in grp[f]]
mask = ((y == 2) & np.isin(src, ['torres', 'funchal'])) | (y < 2)
EDGES = 700 * (10 ** (np.linspace(0, 2595 * np.log10(1 + 24000 / 700), 34) / 2595) - 1); CENTERS = EDGES[1:17]
X = X0.copy()
for s in np.unique(src[mask]): k = mask & (src == s); X[k] = (X0[k] - X0[k].mean(0)) / (X0[k].std(0) + 1e-6)
def weights(idx, yy):
    ss = src[idx]; w = np.ones(len(idx))
    for c in np.unique(yy):
        mc = yy == c; U = np.unique(ss[mc])
        for s in U: k = mc & (ss == s); w[k] = 1 / k.sum() / len(U)
    return w / w.mean()
svm = lambda: SVC(C=1.0, gamma='scale', probability=True, random_state=11)
idx = np.flatnonzero(mask); P = np.zeros((len(y), 3)); rng = np.random.default_rng(0); perm = np.zeros((16, len(y)))
for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=11).split(X[idx], y[idx], grp[idx]):
    tr, te = idx[a], idx[b]; assert not set(grp[tr]) & set(grp[te])
    y1 = (y[tr] > 0).astype(int); m1 = svm().fit(X[tr], y1, sample_weight=weights(tr, y1))
    mos = tr[y[tr] > 0]; y2 = (y[mos] == 2).astype(int); m2 = svm().fit(X[mos], y2, sample_weight=weights(mos, y2))
    def tern(Z): p_m = m1.predict_proba(Z)[:, 1]; p_a = m2.predict_proba(Z)[:, 1]; return np.c_[1 - p_m, p_m * (1 - p_a), p_m * p_a]
    P[te] = tern(X[te])
    for band in range(16):
        Z = X[te].copy(); c = [band + 16 * k for k in range(5)]; Z[:, c] = Z[rng.permutation(len(te))][:, c]; perm[band, te] = tern(Z)[:, 2]
i = idx; yt = y[i]; pa = P[i, 2]; ae = yt == 2; fun = src[i] == 'funchal'
pred = np.where(pa > .5, 2, P[i, :2].argmax(1)); cm = confusion_matrix(yt, pred, labels=[0, 1, 2])
fpr, tpr, _ = roc_curve(ae, pa); keep = np.unique(np.linspace(0, len(fpr) - 1, 120).round().astype(int))
ts = np.round(np.arange(.05, .96, .05), 2); hb = np.linspace(0, 1, 21)
base = roc_auc_score(ae, pa); base_f = roc_auc_score(ae[fun], pa[fun])
curvas = dict(modelo='SVM RBF, dois estágios', classes=['ruido', 'outro_mosquito', 'aedes'], matriz_confusao=cm.tolist(), auroc=float(base),
              auroc_aedes_x_culex=float(base_f),
              roc=dict(fpr=fpr[keep].round(4).tolist(), tpr=tpr[keep].round(4).tolist()),
              por_limiar=[dict(limiar=float(t), precisao=float(ae[pa > t].mean()) if (pa > t).any() else None, recall=float((pa[ae] > t).mean()),
                               fp_outros=float((pa[yt == 1] > t).mean()), fp_ruido=float((pa[yt == 0] > t).mean())) for t in ts],
              histograma=dict(bins=hb.round(2).tolist(), aedes=np.histogram(pa[ae], hb)[0].tolist(), outros=np.histogram(pa[yt == 1], hb)[0].tolist(), ruido=np.histogram(pa[yt == 0], hb)[0].tolist()))
interp = dict(modelo='SVM RBF, dois estágios', auroc_base=round(float(base), 3), auroc_base_aedes_x_culex=round(float(base_f), 3),
              bandas=[dict(centro_hz=round(float(CENTERS[b])), faixa_hz=[round(float(EDGES[b])), round(float(EDGES[b + 2]))],
                           queda_auroc=round(float(base - roc_auc_score(ae, perm[b, i])), 4),
                           queda_auroc_aedes_x_culex=round(float(base_f - roc_auc_score(ae[fun], perm[b, i][fun])), 4)) for b in range(16)],
              probe_origem_outros_mosquitos=json.loads((OUT / 'interpretabilidade_final.json').read_text())['probe_origem_outros_mosquitos'])
(OUT / 'curvas_svm.json').write_text(json.dumps(curvas, indent=1, ensure_ascii=False)); (OUT / 'interpretabilidade_svm.json').write_text(json.dumps(interp, indent=1, ensure_ascii=False))
print('matriz', cm.tolist(), 'auroc', round(base, 3), 'aedes×culex', round(base_f, 3))
print('limiar 0,5:', [d for d in curvas['por_limiar'] if d['limiar'] in (0.2, 0.5)])
print('bandas', [(b['centro_hz'], b['queda_auroc']) for b in interp['bandas']])
