"""MLP e CNN (PyTorch) na mesma receita do modelo final: Aedes só de Torres + Funchal; normalização e pesos por fonte;
dois estágios; validação cruzada por grupo (5 dobras, 3 repetições). Comparação com as árvores."""
from pathlib import Path
import json, numpy as np, torch, torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
OUT = Path(__file__).resolve().parent
torch.set_num_threads(8)
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

def mlp():   # 80 → 128 → 64 → 1
    return nn.Sequential(nn.Linear(80, 128), nn.ReLU(), nn.Dropout(0.3),
                         nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
class CNN(nn.Module):   # entrada: 5 resumos (canais) × 16 faixas de frequência
    def __init__(s):
        super().__init__()
        s.conv = nn.Sequential(nn.Conv1d(5, 32, 3, padding=1), nn.ReLU(),
                               nn.Conv1d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
                               nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveMaxPool1d(1))
        s.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(s, x): return s.head(s.conv(x.view(-1, 5, 16)))

def fit_net(kind, X, yy, w, seed, epochs=60):
    torch.manual_seed(seed); net = mlp() if kind == 'mlp' else CNN()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    Xt, yt, wt = map(lambda a: torch.tensor(a, dtype=torch.float32), (X, yy, w))
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        net.train()
        for ix in torch.randperm(len(Xt), generator=g).split(128):
            loss = (nn.functional.binary_cross_entropy_with_logits(net(Xt[ix]).squeeze(1), yt[ix], reduction='none') * wt[ix]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return lambda Z: torch.sigmoid(net(torch.tensor(Z, dtype=torch.float32)).squeeze(1)).detach().numpy()
hgb = lambda s: HistGradientBoostingClassifier(learning_rate=.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0, min_samples_leaf=12, random_state=s)
et = lambda s: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=8, random_state=s)

def stage(kind, X, idx, yy, te, s, etapa1=False):
    w = weights(idx, yy)
    if kind == 'arvores' and etapa1:   # igual ao modelo final: etapa 1 só com HistGB
        return hgb(s).fit(X[idx], yy, sample_weight=w).predict_proba(X[te])[:, 1]
    if kind == 'arvores':
        return (hgb(s).fit(X[idx], yy, sample_weight=w).predict_proba(X[te])[:, 1] + et(s).fit(X[idx], yy, sample_weight=w).predict_proba(X[te])[:, 1]) / 2
    return fit_net(kind, X[idx], yy, w, s)(X[te])

X = norm(X0); idx_all = np.flatnonzero(mask); res = {}
for kind in ['arvores', 'mlp', 'cnn']:
    runs = []
    for s in [11, 22, 33]:
        P = np.zeros(len(y))
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X[idx_all], y[idx_all], grp[idx_all]):
            tr, te = idx_all[a], idx_all[b]; assert not set(grp[tr]) & set(grp[te])
            y1 = (y[tr] > 0).astype(float); p_m = stage(kind, X, tr, y1, te, s, etapa1=True)
            mos = tr[y[tr] > 0]; y2 = (y[mos] == 2).astype(float); p_a = stage(kind, X, mos, y2, te, s)
            P[te] = p_m * p_a
        i = idx_all; ae = y[i] == 2; call = P[i] > .5; fun = src[i] == 'funchal'
        runs.append(dict(auroc=roc_auc_score(ae, P[i]), precisao=ae[call].mean(), deteccao=call[ae].mean(),
                         outros_alarme=call[y[i] == 1].mean(), ruido_alarme=call[y[i] == 0].mean(),
                         auroc_aedes_x_culex=roc_auc_score(ae[fun], P[i][fun])))
    res[kind] = {k: [round(float(np.mean([r[k] for r in runs])), 3), round(float(np.std([r[k] for r in runs])), 3)] for k in runs[0]}
    print(kind, json.dumps(res[kind]), flush=True)
res['parametros'] = dict(mlp=int(sum(p.numel() for p in mlp().parameters())), cnn=int(sum(p.numel() for p in CNN().parameters())))
(OUT / 'redes_neurais.json').write_text(json.dumps(res, indent=1, ensure_ascii=False)); print(res['parametros'])
