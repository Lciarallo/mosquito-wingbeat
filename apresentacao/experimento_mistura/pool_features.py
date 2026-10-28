"""Extrai atributos HTK (80) das fontes externas e monta o dataset único por arquivo/grupo."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, torch
from scipy.signal import resample_poly
from train_torres_aedes_transfer import bank, describe
from ternary_protocol import SEGMENTS
OUT = Path(__file__).resolve().parent; OLD = ROOT / 'results/torres_aedes_transfer_2026_10_15'
torch.set_num_threads(8)
def extract(frame):
    front, keep, _ = bank('htk'); values = np.zeros((len(frame), len(keep) * 5), np.float32)
    for bucket in ['long', 'short']:
        ids = np.flatnonzero(frame.bucket.eq(bucket).to_numpy())
        if not len(ids): continue
        waves = np.load(ROOT / f'data/cache/reliability_2026_10_01/{bucket}_8k.npy', mmap_mode='r')
        for s in range(0, len(ids), 128):
            ix = ids[s:s + 128]; raw = np.asarray(waves[frame.iloc[ix].wave_index.to_numpy(int)], np.float32)
            with torch.inference_mode(): db = front(torch.from_numpy(resample_poly(raw, 6, 1, axis=1).copy())).numpy()
            values[ix] = describe(db, keep)
    return values
seg = pd.read_csv(SEGMENTS, low_memory=False)
ext = seg[seg.domain.isin(['figshare', 'mosquitosong', 'ufrgs'])].drop_duplicates(['path', 'start_s', 'bucket']).reset_index(drop=True)
Xw = extract(ext)
ext['y'] = np.where(ext.label.eq('Aedes aegypti'), 2, 1)
parts = ext.path.str.split('/')
ext['grp'] = np.select([ext.domain.eq('figshare'), ext.domain.eq('mosquitosong'), ext.domain.eq('ufrgs')],
                       ['funchal:' + parts.str[5] + '/' + parts.str[6], 'mosquitosong:' + ext.group.astype(str), 'ufrgs:' + ext.group.astype(str)])
# um exemplo por arquivo (média das janelas)
F = pd.DataFrame(Xw); F['path'] = ext.path
Xf = F.groupby('path', sort=True).mean()
meta = ext.groupby('path', sort=True).agg(y=('y', 'first'), grp=('grp', 'first'), domain=('domain', 'first')).loc[Xf.index]
meta['domain'] = meta.domain.replace({'figshare': 'funchal'})
P = pd.read_csv(OLD / 'partitions.csv'); prim = P[P.origin == 'primary'].reset_index(drop=True)
with np.load(OLD / 'features_htk.npz') as z: Xp, Xt = z['primary'], z['torres']
X = np.vstack([Xp, Xt, Xf.to_numpy(np.float32)])
y = np.r_[prim.target.to_numpy(), np.full(300, 2), meta.y.to_numpy()]
src = np.r_[prim.domain.to_numpy(), ['torres'] * 300, meta.domain.to_numpy()]
grp = np.r_[prim.domain.to_numpy() + ':' + prim.pcm_block.to_numpy(), [f'torres:{i // 10}' for i in range(300)], meta.grp.to_numpy()]
np.savez_compressed(OUT / 'pool.npz', X=X, y=y, src=src, grp=grp)
df = pd.DataFrame(dict(src=src, y=y, grp=grp))
print(df.groupby(['src', 'y']).agg(n=('grp', 'size'), grupos=('grp', 'nunique')))
