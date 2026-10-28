"""Retesta os demais métodos do projeto na mesma receita do modelo final: Aedes só de Torres + Funchal; normalização e
pesos por fonte; dois estágios; validação cruzada por grupo (5 dobras, 3 repetições)."""
from pathlib import Path
import json, sys, numpy as np, torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
OUT = Path(__file__).resolve().parent; torch.set_num_threads(8)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # GPU (ROCm) quando houver
torch.backends.cudnn.enabled = False   # a LSTM do MIOpen falha na RX 9070 XT; usa a implementação nativa do PyTorch na GPU
z = np.load(OUT / 'pool.npz', allow_pickle=True); X0, y, src, grp = z['X'], z['y'], z['src'], z['grp'].astype(object)
S0 = np.load(OUT / 'seq.npy')
f = src == 'funchal'; grp[f] = ['funchal:' + ('8k' if '8khz' in g else '44k' if '44.1khz' in g else '48k') for g in grp[f]]
mask = ((y == 2) & np.isin(src, ['torres', 'funchal'])) | (y < 2)
FONTES = sorted(np.unique(src[mask])); fid = np.array([FONTES.index(s) if s in FONTES else -1 for s in src])
def norm(A):
    An = A.copy()
    for s in FONTES:
        k = mask & (src == s); ax = tuple(range(A.ndim - 1))
        An[k] = (A[k] - A[k].mean(ax[:1] if A.ndim == 2 else (0, 1))) / (A[k].std(ax[:1] if A.ndim == 2 else (0, 1)) + 1e-6)
    return An
def coral(A):   # CORAL: branqueia cada fonte e recolore com a covariância comum (todas as fontes juntas)
    root = lambda C, e: (lambda w, v: (v * np.clip(w, 1e-6, None) ** e) @ v.T)(*np.linalg.eigh(C))
    centr = np.concatenate([A[mask & (src == s)] - A[mask & (src == s)].mean(0) for s in FONTES])
    alvo = root(LedoitWolf().fit(centr).covariance_, .5); An = A.copy()
    for s in FONTES:
        k = mask & (src == s); Z = A[k] - A[k].mean(0)
        An[k] = Z @ root(LedoitWolf().fit(Z).covariance_, -.5) @ alvo
    return An
def weights(idx, yy):
    ss = src[idx]; w = np.ones(len(idx))
    for c in np.unique(yy):
        mc = yy == c; U = np.unique(ss[mc])
        for s in U: k = mc & (ss == s); w[k] = 1 / k.sum() / len(U)
    return w / w.mean()
class Rec(nn.Module):   # LSTM / Bi-LSTM: lê 32 instantes × 16 faixas
    def __init__(s, bi):
        super().__init__(); s.rnn = nn.LSTM(16, 64, num_layers=2, batch_first=True, dropout=0.2, bidirectional=bi)
        s.out = nn.Sequential(nn.Dropout(0.3), nn.Linear(128 if bi else 64, 1))
    def forward(s, x): h, _ = s.rnn(x); return s.out(h[:, -1])
class GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam = lam; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.lam * g, None
class DANN(nn.Module):  # MLP + cabeça adversária que tenta adivinhar a fonte
    def __init__(s):
        super().__init__(); s.f = nn.Sequential(nn.Linear(80, 128), nn.ReLU(), nn.Dropout(.3), nn.Linear(128, 64), nn.ReLU())
        s.y = nn.Linear(64, 1); s.d = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, len(FONTES)))
    def forward(s, x, lam=0.3): h = s.f(x); return s.y(h), s.d(GradRev.apply(h, lam))
def fit_torch(kind, A, yy, w, dom, seed, epochs):
    torch.manual_seed(seed); net = (Rec(kind == 'bilstm') if kind in ('lstm', 'bilstm') else DANN()).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    At, yt, wt, dt = (torch.tensor(A).to(DEV), torch.tensor(yy, dtype=torch.float32).to(DEV), torch.tensor(w, dtype=torch.float32).to(DEV), torch.tensor(dom).to(DEV))
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        net.train()
        for ix in torch.randperm(len(At), generator=g).split(128):
            ix = ix.to(DEV)
            if kind == 'dann':
                o, od = net(At[ix]); loss = (nn.functional.binary_cross_entropy_with_logits(o.squeeze(1), yt[ix], reduction='none') * wt[ix]).mean() + nn.functional.cross_entropy(od, dt[ix])
            else:
                loss = (nn.functional.binary_cross_entropy_with_logits(net(At[ix]).squeeze(1), yt[ix], reduction='none') * wt[ix]).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 3); opt.step()
    net.eval()
    def pred(B):
        with torch.no_grad(): o = net(torch.tensor(B).to(DEV)); o = o[0] if isinstance(o, tuple) else o
        return torch.sigmoid(o.squeeze(1)).cpu().numpy()
    return pred
hgb = lambda s: HistGradientBoostingClassifier(learning_rate=.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0, min_samples_leaf=12, random_state=s)
et = lambda s: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=8, random_state=s)
def stage(kind, A, idx, yy, te, s, etapa1):
    w = weights(idx, yy)
    if kind == 'logistica': return LogisticRegression(C=0.5, max_iter=5000).fit(A[idx], yy, sample_weight=w).predict_proba(A[te])[:, 1]
    if kind == 'svm': return SVC(C=1.0, gamma='scale', probability=True, random_state=s).fit(A[idx], yy, sample_weight=w).predict_proba(A[te])[:, 1]
    if kind == 'coral_arvores':
        if etapa1: return hgb(s).fit(A[idx], yy, sample_weight=w).predict_proba(A[te])[:, 1]
        return (hgb(s).fit(A[idx], yy, sample_weight=w).predict_proba(A[te])[:, 1] + et(s).fit(A[idx], yy, sample_weight=w).predict_proba(A[te])[:, 1]) / 2
    ep = 40 if kind in ('lstm', 'bilstm') else 60
    return fit_torch(kind, A[idx].astype(np.float32), yy, w, np.clip(fid[idx], 0, None), s, ep)(A[te].astype(np.float32))
DATA = {'logistica': norm(X0), 'svm': norm(X0), 'dann': norm(X0), 'coral_arvores': coral(X0), 'lstm': norm(S0), 'bilstm': norm(S0)}
kinds = sys.argv[1:] or list(DATA)
res = json.loads((OUT / 'metodos_restantes.json').read_text()) if (OUT / 'metodos_restantes.json').exists() else {}
idx_all = np.flatnonzero(mask)
for kind in kinds:
    A = DATA[kind]; runs = []
    for s in [11, 22, 33]:
        P = np.zeros(len(y))
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X0[idx_all], y[idx_all], grp[idx_all]):
            tr, te = idx_all[a], idx_all[b]; assert not set(grp[tr]) & set(grp[te])
            y1 = (y[tr] > 0).astype(float); p_m = stage(kind, A, tr, y1, te, s, True)
            mos = tr[y[tr] > 0]; y2 = (y[mos] == 2).astype(float); p_a = stage(kind, A, mos, y2, te, s, False)
            P[te] = p_m * p_a
        i = idx_all; ae = y[i] == 2; call = P[i] > .5; fun = src[i] == 'funchal'
        runs.append(dict(auroc=roc_auc_score(ae, P[i]), precisao=float(ae[call].mean()) if call.any() else 0.0, deteccao=call[ae].mean(),
                         outros_alarme=call[y[i] == 1].mean(), ruido_alarme=call[y[i] == 0].mean(), auroc_aedes_x_culex=roc_auc_score(ae[fun], P[i][fun])))
    res[kind] = {k: [round(float(np.mean([r[k] for r in runs])), 3), round(float(np.std([r[k] for r in runs])), 3)] for k in runs[0]}
    print(kind, json.dumps(res[kind]), flush=True)
    atual = json.loads((OUT / 'metodos_restantes.json').read_text()) if (OUT / 'metodos_restantes.json').exists() else {}
    atual[kind] = res[kind]; (OUT / 'metodos_restantes.json').write_text(json.dumps(atual, indent=1, ensure_ascii=False))
