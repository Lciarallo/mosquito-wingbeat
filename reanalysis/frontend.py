"""Reanalysis v2 - common frontend (exactly as declared in protocol_v2.json).

Every source goes through the same path: decode -> mono -> 8 kHz -> 100-3400 Hz
band-pass -> 1 s windows -> validity rule -> unit-RMS gain -> log-mel 40 x 59
+ tabular statistics + wingbeat descriptors.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
SR = 8000
WIN = SR
MAX_WIN = 30
N_FFT, HOP, N_MELS, FMIN, FMAX = 512, 128, 40, 100.0, 3400.0
SOS = butter(4, [FMIN, FMAX], btype="bandpass", fs=SR, output="sos")
DESCRIPTORS = ["f0_hz", "log_h2_h1", "log_h3_h1", "log_h4_h1", "centroid_hz",
               "bandwidth_hz", "flatness", "frac_below_1k", "log_f0_salience"]


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank():
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    edges = mel_to_hz(np.linspace(hz_to_mel(FMIN), hz_to_mel(FMAX), N_MELS + 2))
    fb = np.zeros((N_MELS, len(freqs)))
    for i in range(N_MELS):
        lo, c, hi = edges[i:i + 3]
        fb[i] = np.clip(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0, None)
    return fb


FB = mel_filterbank()
HANN = np.hanning(N_FFT + 1)[:-1]


def decode(path):
    try:
        x, rate = sf.read(str(path), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
    except Exception:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-f", "f32le", "-"],
                             capture_output=True, check=True).stdout
        info = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                               "stream=sample_rate", "-of", "csv=p=0", str(path)],
                              capture_output=True, text=True, check=True).stdout
        x, rate = np.frombuffer(raw, dtype="<f4").copy(), int(info.strip().split(",")[0])
    if rate != SR:
        g = np.gcd(int(rate), SR)
        x = resample_poly(x, SR // g, int(rate) // g).astype("float32")
    return x


def windows(x):
    """Returns (list of raw 1 s windows, start seconds, count of rejected windows)."""
    if len(x) < int(0.95 * WIN):
        return [], [], 0
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)), mode="reflect")
    y = sosfiltfilt(SOS, x).astype("float32") if len(x) > 30 else x
    starts = range(0, len(y) - WIN + 1, WIN)
    valid, t, rejected = [], [], 0
    for s in starts:
        w = x[s:s + WIN]
        rms = float(np.sqrt(np.mean(w.astype("float64") ** 2)))
        if rms < 1e-4 or np.mean(np.abs(w) >= 0.999) > 0.01:
            rejected += 1
            continue
        valid.append(y[s:s + WIN]); t.append(s / SR)
    if len(valid) > MAX_WIN:
        keep = np.unique(np.linspace(0, len(valid) - 1, MAX_WIN).round().astype(int))
        valid = [valid[i] for i in keep]; t = [t[i] for i in keep]
    return valid, t, rejected


def logmel(w):
    frames = np.lib.stride_tricks.sliding_window_view(w, N_FFT)[::HOP]  # 59 frames, no padding
    spec = np.abs(np.fft.rfft(frames * HANN, axis=1)) ** 2
    m = np.log(spec @ FB.T + 1e-6).T  # (40, frames)
    return (m - m.mean()).astype("float32")


def descriptors(w):
    mag = np.abs(np.fft.rfft(w * np.hanning(len(w))))
    f = np.fft.rfftfreq(len(w), 1 / SR)  # 1 Hz resolution
    cand = np.arange(250, 1001)  # amendment A1: 150 Hz floor captured handling/room noise
    def amp(fr):
        i = np.clip(np.round(fr).astype(int), 0, len(mag) - 3)
        return np.max(np.stack([mag[i - 2], mag[i - 1], mag[i], mag[i + 1], mag[i + 2]]), axis=0)
    # Harmonic sum in the log domain (harmonic product spectrum): a plain linear
    # sum locks onto sub-harmonics at the 150 Hz search floor (amendment A1).
    harm = np.exp(sum(np.log(amp(k * cand) + 1e-9) for k in range(1, 4)) / 3)
    f0 = float(cand[np.argmax(harm)])
    h = [float(amp(np.array([k * f0]))[0]) + 1e-9 for k in range(1, 5)]
    p = mag ** 2; band = (f >= FMIN) & (f <= FMAX); pb = p[band] + 1e-12; fb = f[band]
    centroid = float((fb * pb).sum() / pb.sum())
    bw = float(np.sqrt(((fb - centroid) ** 2 * pb).sum() / pb.sum()))
    flat = float(np.exp(np.mean(np.log(pb))) / np.mean(pb))
    low = float(pb[fb < 1000].sum() / pb.sum())
    sal = float(np.log(harm.max() / (np.median(harm) + 1e-9)))
    return [f0, np.log(h[1] / h[0]), np.log(h[2] / h[0]), np.log(h[3] / h[0]), centroid, bw, flat, low, sal]


def process(item):
    idx, path = item
    try:
        x = decode(ROOT / path)
        ws, t, rej = windows(x)
        mels, tab = [], []
        for w in ws:
            w = w / max(float(np.sqrt(np.mean(w ** 2))), 1e-12)
            m = logmel(w)
            mels.append(m)
            tab.append(np.r_[m.mean(1), m.std(1), descriptors(w)].astype("float32"))
        return idx, mels, tab, t, rej, None
    except Exception as e:  # recorded, never silently dropped
        return idx, [], [], [], 0, repr(e)


def main(workers=20):
    man = pd.read_csv(ROOT / "results/v2/manifest_files.csv")
    out = ROOT / "data/cache/v2"; out.mkdir(parents=True, exist_ok=True)
    rows, mels, tabs, errors, per_file = [], [], [], {}, []
    with ProcessPoolExecutor(workers) as ex:
        for n, (i, m, tb, t, rej, err) in enumerate(ex.map(process, enumerate(man.path), chunksize=4), 1):
            if err: errors[man.path[i]] = err
            per_file.append(dict(file_index=i, windows=len(m), rejected=rej, error=err))
            for k in range(len(m)):
                rows.append(dict(file_index=i, start_s=t[k])); mels.append(m[k]); tabs.append(tb[k])
            if n % 300 == 0: print("files", n, "/", len(man), flush=True)
    win = pd.DataFrame(rows)
    X = np.stack(mels); T = np.stack(tabs)
    np.save(out / "logmel.npy", X); np.save(out / "tabular.npy", T)
    win.to_csv(out / "windows.csv", index=False)
    pd.DataFrame(per_file).to_csv(out / "per_file.csv", index=False)
    names = [f"mel{b}_mean" for b in range(N_MELS)] + [f"mel{b}_std" for b in range(N_MELS)] + DESCRIPTORS
    digest = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    meta = dict(n_windows=len(win), logmel_shape=list(X.shape), tabular_names=names, errors=errors,
                sha256={p.name: digest(p) for p in [out / "logmel.npy", out / "tabular.npy", out / "windows.csv"]},
                frontend_sha256=digest(Path(__file__)), finite=bool(np.isfinite(X).all() and np.isfinite(T).all()))
    json.dump(meta, open(out / "cache_meta.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in meta.items() if k != "tabular_names"}, indent=1)[:1500])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
