"""v4 augmented copy of every development file (protocol_v4.json, representations.augmented).

One copy per development file, deterministic from (augmentation seed, file_id). Only external
SLR28 RIRs/noises are mixed in; no ESC-50, Pete or HumBug audio. Features go through the unchanged
v3 code path (band-pass, whitening, Perch chunks, one-second tabular vectors) and only the whitened
versions used by v4 are kept. Resumable by blocks; the reserved v3 test is never decoded.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt, sosfiltfilt

from reanalysis import frontend as fe
from reanalysis.v3_embed import whiten
from reanalysis.v3_features import MODEL, acoustic_features, audio_chunks, digest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/v3_continuation/manifest.csv"
SLR28 = ROOT / "data/external/openslr28/extracted/RIRS_NOISES"
OUT = ROOT / "data/cache/v4_augmented"
SEED = 20261102
DEV_ROLES = ["train", "noise_train"]


def slr28_files(root=SLR28):
    real = sorted((root / "real_rirs_isotropic_noises").glob("*.wav"))
    rirs = [p for p in real if "_rir_" in p.name]
    noises = [p for p in real if "noise" in p.name] + sorted((root / "pointsource_noises").glob("*.wav"))
    if not rirs or not noises:
        raise FileNotFoundError(f"SLR28 RIRs/noises not found under {root}")
    return rirs, noises


def read_8k(path):
    x, rate = sf.read(str(path), dtype="float32", always_2d=True)
    x = x[:, 0]
    if rate != fe.SR:
        g = np.gcd(int(rate), fe.SR)
        x = resample_poly(x, fe.SR // g, int(rate) // g).astype("float32")
    return x


def peaking(f0, gain_db, q, sr=fe.SR):
    """RBJ audio-EQ-cookbook peaking biquad as one second-order section."""
    a = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / sr
    alpha = np.sin(w) / (2 * q)
    b = np.array([1 + alpha * a, -2 * np.cos(w), 1 - alpha * a])
    den = np.array([1 + alpha / a, -2 * np.cos(w), 1 - alpha / a])
    return np.r_[b / den[0], den / den[0]][None, :]


def augment(x, rng, rirs, noises):
    """Microphone response, optional real RIR, optional external noise, gain and clipping."""
    params = {}
    hp, lp = rng.uniform(100, 600), rng.uniform(2000, 3400)
    sos = [butter(2, hp, "highpass", fs=fe.SR, output="sos"), butter(4, lp, "lowpass", fs=fe.SR, output="sos")]
    peaks = [(float(np.exp(rng.uniform(np.log(150), np.log(3000)))), float(rng.uniform(-9, 9)), float(rng.uniform(.7, 3)))
             for _ in range(2)]
    sos += [peaking(*p) for p in peaks]
    y = sosfilt(np.vstack(sos), x.astype("float64"))
    params.update(highpass_hz=float(hp), lowpass_hz=float(lp), peaks=peaks)
    if rng.random() < .5:
        path = rirs[rng.integers(len(rirs))]
        h = read_8k(path)
        h = h / (np.sqrt(np.sum(h.astype("float64") ** 2)) + 1e-12)
        y = fftconvolve(y, h)[: len(y)]
        params["rir"] = path.name
    if rng.random() < .8:
        path = noises[rng.integers(len(noises))]
        n = read_8k(path).astype("float64")
        if len(n) < len(y):
            n = np.tile(n, int(np.ceil(len(y) / max(len(n), 1))))
        start = rng.integers(0, len(n) - len(y) + 1)
        n = n[start:start + len(y)]
        snr = rng.uniform(0, 20)
        ps, pn = np.mean(y ** 2) + 1e-12, np.mean(n ** 2) + 1e-12
        y = y + n * np.sqrt(ps / (pn * 10 ** (snr / 10)))
        params.update(noise=path.name, snr_db=float(snr))
    peak = np.max(np.abs(y)) + 1e-12
    gain_db = rng.uniform(-20, 6)
    y = np.clip(y / peak * 10 ** (gain_db / 20), -1, 1)
    params["gain_db_rel_peak"] = float(gain_db)
    return y.astype("float32"), params


def file_rng(file_id, copy=0, seed=SEED):
    return np.random.default_rng([seed, int(file_id), copy])


def prepare(item, rirs, noises):
    file_id, path = item
    x = fe.decode(ROOT / path)
    if len(x) == 0 or not np.isfinite(x).all():
        raise ValueError(f"Empty or non-finite audio: {path}")
    y, params = augment(x, file_rng(file_id), rirs, noises)
    if len(y) > 30:
        y = sosfiltfilt(fe.SOS, y).astype("float32")
    white = whiten(y)
    return int(file_id), audio_chunks(white), acoustic_features(white), params


def development_items(manifest=MANIFEST):
    man = pd.read_csv(manifest)
    dev = man[man.role.isin(DEV_ROLES)]
    if dev.role.isin(["locked_test", "locked_noise", "quarantine_holdout_related"]).any():
        raise AssertionError("reserved rows selected for augmentation")
    return list(zip(dev.file_id, dev.path))


def main(out=OUT, block_size=64, threads=8, batch_size=16, limit=None):
    import onnxruntime as ort
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rirs, noises = slr28_files()
    items = development_items()[:limit]
    identity = dict(manifest_sha256=digest(MANIFEST), model_sha256=digest(MODEL), code_sha256=digest(__file__),
                    v3_features_sha256=digest(ROOT / "reanalysis/v3_features.py"),
                    frontend_sha256=digest(fe.__file__), slr28_zip_sha256=(ROOT / "data/external/openslr28/rirs_noises.zip.sha256").read_text().split()[0],
                    n_rirs=len(rirs), n_noises=len(noises), seed=SEED, copy=0,
                    file_ids=[int(i) for i, _ in items], block_size=block_size)
    identity_path = out / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Existing augmented cache has a different identity")
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
    session = ort.InferenceSession(str(MODEL), options, providers=["CPUExecutionProvider"])
    started = time.monotonic()
    for start in range(0, len(items), block_size):
        block = items[start:start + block_size]
        destination = out / f"block_{start:06d}.npz"
        if destination.exists():
            continue
        with ThreadPoolExecutor(4) as executor:
            prepared = list(executor.map(lambda item: prepare(item, rirs, noises), block))
        chunks = [c for row in prepared for c in row[1]]
        counts = [len(row[1]) for row in prepared]
        embeddings = np.concatenate([session.run(["embedding"], {"inputs": np.stack(chunks[i:i + batch_size])})[0]
                                     for i in range(0, len(chunks), batch_size)])
        cursor, vectors = 0, []
        for count in counts:
            vectors.append(embeddings[cursor:cursor + count].mean(0))
            cursor += count
        arrays = dict(file_ids=np.array([row[0] for row in prepared], dtype=np.int64),
                      perch_white=np.asarray(vectors, dtype="float32"),
                      tabular_white=np.stack([row[2] for row in prepared]).astype("float32"))
        for value in arrays.values():
            if not np.isfinite(value).all():
                raise ValueError(f"Non-finite augmented features in block {start}")
        np.savez_compressed(destination.with_suffix(".tmp.npz"), **arrays)
        destination.with_suffix(".tmp.npz").replace(destination)
        with open(out / "params.jsonl", "a") as log:
            for row in prepared:
                log.write(json.dumps(dict(file_id=row[0], **row[3])) + "\n")
        print(json.dumps(dict(completed=min(start + block_size, len(items)), total=len(items),
                              elapsed_seconds=round(time.monotonic() - started, 1))), flush=True)
    blocks = [np.load(out / f"block_{s:06d}.npz") for s in range(0, len(items), block_size)]
    joined = {key: np.concatenate([b[key] for b in blocks]) for key in blocks[0].files}
    np.savez_compressed(out / "features.tmp.npz", **joined)
    (out / "features.tmp.npz").replace(out / "features.npz")
    meta = {k: v for k, v in identity.items() if k != "file_ids"}
    meta.update(n_files=len(items), features_sha256=digest(out / "features.npz"), complete=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    main(threads=args.threads, limit=args.limit)
