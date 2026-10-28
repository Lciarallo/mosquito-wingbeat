"""Sequências no tempo (32 instantes × 16 faixas mel) alinhadas linha a linha com pool.npz, para LSTM/Bi-LSTM.
Confere o alinhamento recalculando os 80 atributos a partir do mesmo espectrograma."""
from pathlib import Path
import sys, json
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'results/torres_aedes_transfer_2026_10_15/source'))
import numpy as np, pandas as pd, torch
from scipy.signal import resample_poly
from train_torres_aedes_transfer import bank, describe
from ternary_protocol import SEGMENTS
OUT = Path(__file__).resolve().parent; OLD = ROOT / 'results/torres_aedes_transfer_2026_10_15'; T = 32
torch.set_num_threads(8); front, keep, _ = bank('htk')
def seq_of(db):   # db: (n, frames, 32) → (n, T, 16) em dB relativo, como em describe()
    z = db[:, 2:-2, keep]; z = np.clip(z - z.max((1, 2), keepdims=True), -80, 0)
    edges = np.linspace(0, z.shape[1], T + 1).round().astype(int)
    return np.stack([z[:, a:max(b, a + 1)].mean(1) for a, b in zip(edges[:-1], edges[1:])], 1).astype(np.float32)
def from_cache(frame):
    seqs = np.zeros((len(frame), T, 16), np.float32); feats = np.zeros((len(frame), 80), np.float32)
    for bucket in ['long', 'short']:
        ids = np.flatnonzero(frame.bucket.eq(bucket).to_numpy()) if 'bucket' in frame else np.arange(len(frame))
        if bucket == 'short' and 'bucket' not in frame: break
        if not len(ids): continue
        waves = np.load(ROOT / f'data/cache/reliability_2026_10_01/{bucket}_8k.npy', mmap_mode='r')
        for s in range(0, len(ids), 128):
            ix = ids[s:s + 128]; raw = np.asarray(waves[frame.iloc[ix].wave_index.to_numpy(int)], np.float32)
            with torch.inference_mode(): db = front(torch.from_numpy(resample_poly(raw, 6, 1, axis=1).copy())).numpy()
            seqs[ix] = seq_of(db); feats[ix] = describe(db, keep)
    return seqs, feats
# 1) corpus original: 3 janelas por grupo, média por grupo (mesma regra dos 80 atributos)
P = pd.read_csv(OLD / 'partitions.csv'); prim = P[P.origin == 'primary'].reset_index(drop=True)
win = pd.read_csv(OLD / 'selected_windows.csv'); sw, fw = from_cache(win)
order = {g: i for i, g in enumerate(prim.group)}; gi = win.group.map(order).to_numpy()
Sp = np.zeros((len(prim), T, 16), np.float32); Fp = np.zeros((len(prim), 80), np.float32); n = np.zeros(len(prim))
np.add.at(Sp, gi, sw); np.add.at(Fp, gi, fw); np.add.at(n, gi, 1); Sp /= n[:, None, None]; Fp /= n[:, None]
# 2) Torres: tensores mel dos autores (300 × 188 × 32)
raw = np.asarray(json.loads((ROOT / 'data/external/torres_2025/Dataset/Melspec.json').read_text())['mspec'], np.float32)
St = seq_of(raw); Ft = describe(raw, keep)
# 3) externos: Funchal, MosquitoSong, UFRGS — média das janelas por arquivo, arquivos em ordem alfabética
seg = pd.read_csv(SEGMENTS, low_memory=False)
ext = seg[seg.domain.isin(['figshare', 'mosquitosong', 'ufrgs'])].drop_duplicates(['path', 'start_s', 'bucket']).reset_index(drop=True)
se, fe = from_cache(ext)
paths = sorted(ext.path.unique()); pi = ext.path.map({p: i for i, p in enumerate(paths)}).to_numpy()
Se = np.zeros((len(paths), T, 16), np.float32); Fe = np.zeros((len(paths), 80), np.float32); m = np.zeros(len(paths))
np.add.at(Se, pi, se); np.add.at(Fe, pi, fe); np.add.at(m, pi, 1); Se /= m[:, None, None]; Fe /= m[:, None]
S = np.concatenate([Sp, St, Se]); F = np.concatenate([Fp, Ft, Fe])
pool = np.load(OUT / 'pool.npz', allow_pickle=True)['X']
err = np.abs(F - pool).max(); print('linhas', S.shape, '| erro máximo ao recalcular os 80 atributos:', float(err))
assert S.shape[0] == pool.shape[0] and err < 1e-2, 'desalinhado'
np.save(OUT / 'seq.npy', S); print('salvo seq.npy')
