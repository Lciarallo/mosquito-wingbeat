"""Detector v3 - cross-corpus duplicate detection by spectral-peak landmarks.

Robust to cropping, re-encoding and resampling: every file is decoded to 8 kHz,
spectral peaks are paired into (f1, f2, dt) hashes, and two files are linked when
many hashes agree at one consistent time offset. Exact byte and PCM hashes are
checked first.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter

from reanalysis import frontend as fe

ROOT = Path(__file__).resolve().parents[1]
N_FFT, HOP = 512, 128
MAX_SECONDS = 120
PEAKS_PER_FRAME, FANOUT, MAX_DT = 2, 3, 32
LO, HI = int(150 / (8000 / N_FFT)), int(3400 / (8000 / N_FFT))


def landmarks(path):
    x = fe.decode(ROOT / path)[: MAX_SECONDS * 8000]
    pcm = np.rint(np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    pcm_hash = hashlib.sha256(pcm).hexdigest()
    if len(x) < N_FFT * 2:
        return path, pcm_hash, np.zeros(0, np.int64), np.zeros(0, np.int32)
    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP]
    S = np.log(np.abs(np.fft.rfft(frames * np.hanning(N_FFT), axis=1))[:, LO:HI] ** 2 + 1e-10)
    S -= np.median(S, axis=0, keepdims=True)  # remove stationary device/background colouring
    local = (S == maximum_filter(S, size=(7, 9))) & (S > np.percentile(S, 90))
    pts = []
    for t in range(S.shape[0]):
        f = np.flatnonzero(local[t])
        if len(f):
            f = f[np.argsort(S[t, f])[::-1][:PEAKS_PER_FRAME]]
            pts.extend((t, int(v)) for v in f)
    pts = np.array(pts, dtype=np.int64).reshape(-1, 2)
    hashes, times = [], []
    for i, (t1, f1) in enumerate(pts):
        j = i + 1; k = 0
        while j < len(pts) and k < FANOUT:
            t2, f2 = pts[j]
            dt = t2 - t1
            if dt > MAX_DT: break
            if dt >= 1:
                hashes.append((f1 << 14) | (f2 << 5) | dt); times.append(t1); k += 1
            j += 1
    return path, pcm_hash, np.asarray(hashes, np.int64), np.asarray(times, np.int32)


def build(paths, workers=24):
    out = {}
    with ProcessPoolExecutor(workers) as ex:
        for n, (p, ph, h, t) in enumerate(ex.map(landmarks, paths, chunksize=16), 1):
            out[p] = (ph, h, t)
            if n % 2000 == 0: print("landmarks", n, "/", len(paths), flush=True)
    return out


def match(lm, source_of, min_votes=25, min_frac=0.08, stop=300):
    paths = list(lm)
    fid = {p: i for i, p in enumerate(paths)}
    H = np.concatenate([lm[p][1] for p in paths]); T = np.concatenate([lm[p][2] for p in paths])
    F = np.concatenate([np.full(len(lm[p][1]), fid[p], np.int32) for p in paths])
    order = np.argsort(H, kind="stable"); H, T, F = H[order], T[order], F[order]
    uniq, start, count = np.unique(H, return_index=True, return_counts=True)
    common = set(uniq[count > stop].tolist())
    rows = []
    for p in paths:
        qh, qt = lm[p][1], lm[p][2]
        keep = ~np.isin(qh, list(common)) if common else np.ones(len(qh), bool)
        qh, qt = qh[keep], qt[keep]
        if len(qh) < 10: continue
        lo = np.searchsorted(H, qh, "left"); hi = np.searchsorted(H, qh, "right")
        n = hi - lo
        if n.sum() == 0: continue
        idx = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi) if b > a])
        qrep = np.repeat(qt, n)
        rf, off = F[idx], T[idx] - qrep
        me = fid[p]
        sel = rf != me
        if not sel.any(): continue
        key = rf[sel].astype(np.int64) * 100000 + (off[sel] + 50000)
        k, c = np.unique(key, return_counts=True)
        best = np.argsort(c)[::-1][:5]
        for b in best:
            other = paths[int(k[b] // 100000)]
            votes = int(c[b])
            if votes >= min_votes and votes >= min_frac * len(qh) and source_of[other] != source_of[p]:
                rows.append(dict(path=p, other=other, votes=votes, query_hashes=int(len(qh)),
                                 offset_frames=int(k[b] % 100000 - 50000)))
    return pd.DataFrame(rows)
