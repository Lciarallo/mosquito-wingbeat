"""Experimento didático (retrospectivo): misturar Torres e Funchal ao corpus original.

Pergunta: somar positivos de fontes únicas ensina "Aedes" ou ensina "fonte"?
Compara duas formas de dividir treino/teste com o MESMO modelo:
  A) por clipe/tensor  (Torres e Funchal tratados como exemplos independentes)
  B) por fonte/aparelho (Torres = 1 grupo; Funchal = pasta espécie×taxa de amostragem)
O corpus original sempre usa blocos PCM. Nada aqui é teste cego.
"""
from pathlib import Path
import sys, json
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, torch
from scipy.signal import resample_poly
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.decomposition import PCA
from train_torres_aedes_transfer import bank, describe
from ternary_protocol import SEGMENTS

OUT = Path(__file__).resolve().parent
OLD = ROOT / 'results/torres_aedes_transfer_2026_10_15'
SEED = 20261027
torch.set_num_threads(8)

def extract(frame):  # mesma função de results/torres_generalizacao_2026_10_15/verify_and_external.py
    front, keep, _ = bank('htk'); values = np.zeros((len(frame), len(keep) * 5), np.float32)
    for bucket in ['long', 'short']:
        ids = np.flatnonzero(frame.bucket.eq(bucket))
        if not len(ids): continue
        waves = np.load(ROOT / f'data/cache/reliability_2026_10_01/{bucket}_8k.npy', mmap_mode='r')
        for s in range(0, len(ids), 128):
            ix = ids[s:s + 128]; raw = np.asarray(waves[frame.iloc[ix].wave_index.to_numpy(int)], np.float32)
            with torch.inference_mode(): db = front(torch.from_numpy(resample_poly(raw, 6, 1, axis=1).copy())).numpy()
            values[ix] = describe(db, keep)
    return values

# --- dados
P = pd.read_csv(OLD / 'partitions.csv'); prim = P[P.origin == 'primary'].reset_index(drop=True)
with np.load(OLD / 'features_htk.npz') as z: Xp, Xt = z['primary'], z['torres']
seg = pd.read_csv(SEGMENTS, low_memory=False)
fun = seg[seg.domain.eq('figshare')].drop_duplicates(['path', 'start_s']).reset_index(drop=True)
Xf = extract(fun)
fun['species'] = fun.path.str.split('/').str[5]; fun['folder'] = fun.path.str.split('/').str[6]
assert set(fun.species) == {'Aedes Aegypti', 'Culex Pipiens & Quinquefasctiatus', 'Culiseta'}, set(fun.species)
yf = np.where(fun.species.eq('Aedes Aegypti'), 2, 1)

X = np.vstack([Xp, Xt, Xf]); y = np.r_[prim.target.to_numpy(), np.full(300, 2), yf]
src = np.r_[prim.domain.to_numpy(), ['torres'] * 300, ['funchal'] * len(fun)]
g_prim = prim.pcm_block.to_numpy()
groups = {
    'A_por_clipe': np.r_[g_prim, [f'torres:{i}' for i in range(300)], [f'funchal:{i}' for i in range(len(fun))]],
    'B_por_fonte': np.r_[g_prim, ['torres'] * 300, ('funchal:' + fun.species + '/' + fun.folder).to_numpy()],
}

def weights(yy):
    w = np.ones(len(yy));
    for c in np.unique(yy): w[yy == c] = 1.0 / (yy == c).sum()
    return w / w.mean()

def run(name, g):
    pred = np.full(len(y), -1); sg = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    for tr, te in sg.split(X, y, g):
        assert not set(g[tr]) & set(g[te]), 'vazamento de grupo'
        m = HistGradientBoostingClassifier(learning_rate=.06, max_iter=180, max_leaf_nodes=15, l2_regularization=1.0,
                                           min_samples_leaf=12, random_state=SEED).fit(X[tr], y[tr], sample_weight=weights(y[tr]))
        pred[te] = m.predict(X[te])
    res = {}
    for s in ['dryad', 'humbug', 'torres', 'funchal']:
        a = (src == s) & (y == 2); res[f'aedes_{s}'] = [int((pred[a] == 2).sum()), int(a.sum())]
    a = y == 2; res['aedes_total'] = [int((pred[a] == 2).sum()), int(a.sum())]
    o = (y == 1) & (src == 'funchal'); res['funchal_outros_chamados_aedes'] = [int((pred[o] == 2).sum()), int(o.sum())]
    n = y == 0; res['ruido_chamado_aedes'] = [int((pred[n] == 2).sum()), int(n.sum())]
    return res

out = dict(nota=__doc__.strip(), n=dict(original=len(prim), torres=300, funchal=len(fun),
           funchal_aedes=int((yf == 2).sum()), funchal_pastas=sorted(fun.folder.unique().tolist())))
for k, g in groups.items(): out[k] = run(k, g)

# --- probe de origem entre Aedes (original × Torres × Funchal), chance 1/3
a = y == 2; Xa, sa = X[a], src[a]; oa = np.where(np.isin(sa, ['dryad', 'humbug']), 'original', sa)
ga = groups['A_por_clipe'][a]; pr = np.empty(len(oa), object)
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=7).split(Xa, oa, ga):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight='balanced')).fit(Xa[tr], oa[tr]); pr[te] = m.predict(Xa[te])
out['probe_origem_entre_aedes'] = dict(classes=['original', 'torres', 'funchal'], balanced_accuracy=float(balanced_accuracy_score(oa, pr)), chance=1 / 3,
                                       n={c: int((oa == c).sum()) for c in ['original', 'torres', 'funchal']})

# --- mapa 2D: PCA ajustado em original+Torres (como no slide), Funchal Aedes projetado (amostra de 300)
sc = StandardScaler().fit(np.vstack([Xp, Xt])); pca = PCA(2, random_state=0).fit(sc.transform(np.vstack([Xp, Xt])))
idx = np.random.default_rng(0).choice(np.flatnonzero(yf == 2), 300, replace=False)
Zf = pca.transform(sc.transform(Xf[idx]))
pd.DataFrame(dict(pc1=Zf[:, 0], pc2=Zf[:, 1], grupo='funchal')).to_csv(OUT / 'pca_funchal.csv', index=False)
(OUT / 'resultados_mistura.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps({k: v for k, v in out.items() if k != 'nota'}, indent=1, ensure_ascii=False))
