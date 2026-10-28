"""Resumable, bounded-memory Perch and acoustic features, one vector per file.

Both versions use the identical waveform, chunks and fixed gain. No class label
enters feature extraction. Short recordings are padded with zeros, never repeated.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.signal import resample_poly, sosfiltfilt

from reanalysis import frontend as fe
from reanalysis.v3_embed import whiten

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "data/models/perch_v2_onnx/perch_v2.onnx"
MANIFEST = ROOT / "results/v3_continuation/manifest.csv"
OUT = ROOT / "data/cache/v3_continuation"
CHUNK = 160000


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audio_chunks(x):
    """Same five-second chunk rule as the inherited script, gain RMS 0.05."""
    y = resample_poly(x, 4, 1).astype("float32")
    if len(y) <= CHUNK:
        pad = CHUNK - len(y)
        y = np.pad(y, (pad // 2, pad - pad // 2))
        starts = [0]
    else:
        n = len(y) // CHUNK
        starts = np.unique(np.linspace(0, n - 1, min(n, 6)).round().astype(int)) * CHUNK
    out = []
    for start in starts:
        c = y[start:start + CHUNK]
        rms = float(np.sqrt(np.mean(c.astype("float64") ** 2)))
        out.append((c / max(rms, 1e-8) * 0.05).astype("float32"))
    return out


def acoustic_features(x):
    """Mean of <=30 one-second descriptors; zero-pad a short file once."""
    if len(x) < fe.WIN:
        pad = fe.WIN - len(x)
        x = np.pad(x, (pad // 2, pad - pad // 2))
    n = len(x) // fe.WIN
    starts = np.unique(np.linspace(0, n - 1, min(n, 30)).round().astype(int)) * fe.WIN
    rows = []
    for start in starts:
        w = x[start:start + fe.WIN]
        w = w / max(float(np.sqrt(np.mean(w.astype("float64") ** 2))), 1e-12)
        mel = fe.logmel(w)
        rows.append(np.r_[mel.mean(1), mel.std(1), fe.descriptors(w)])
    return np.mean(rows, axis=0).astype("float32")


def prepare_audio(item):
    file_id, path = item
    x = fe.decode(ROOT / path)
    if len(x) == 0 or not np.isfinite(x).all():
        raise ValueError(f"Empty or non-finite audio: {path}")
    if len(x) > 30:
        x = sosfiltfilt(fe.SOS, x).astype("float32")
    white = whiten(x)
    return int(file_id), audio_chunks(x), audio_chunks(white), acoustic_features(x), acoustic_features(white)


def main(out=OUT, block_size=64, threads=8, batch_size=16, limit=None):
    import onnxruntime as ort
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(MANIFEST)
    selected = man[man.role.ne("quarantine_holdout_related")]
    if limit is not None:
        selected = selected.iloc[:limit]
    identity = dict(manifest_sha256=digest(MANIFEST), model_sha256=digest(MODEL),
                    code_sha256=digest(__file__), frontend_sha256=digest(fe.__file__),
                    whitening_code_sha256=digest(ROOT / "reanalysis/v3_embed.py"),
                    file_ids=selected.file_id.tolist(), block_size=block_size,
                    aggregation="mean embedding per file; mean one-second acoustic vectors per file",
                    input_gain_rms=0.05)
    identity_path = out / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("Existing feature cache has a different immutable identity")
    else:
        identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(MODEL), options, providers=["CPUExecutionProvider"])
    started = time.monotonic()
    items = list(zip(selected.file_id, selected.path))
    for start in range(0, len(items), block_size):
        block = items[start:start + block_size]
        destination = out / f"block_{start:06d}.npz"
        if destination.exists():
            with np.load(destination) as saved:
                if saved["file_ids"].tolist() != [int(i) for i, _ in block]:
                    raise ValueError("Wrong file ordering in checkpoint")
            continue
        # Only this small block is queued: decoding cannot accumulate the whole corpus in RAM.
        with ThreadPoolExecutor(4) as executor:
            prepared = list(executor.map(prepare_audio, block))
        arrays = {"file_ids": np.array([i for i, *_ in prepared], dtype=np.int64)}
        for feature_name, part in [("perch_plain", 1), ("perch_white", 2)]:
            chunks = [chunk for row in prepared for chunk in row[part]]
            counts = [len(row[part]) for row in prepared]
            embeddings = []
            for offset in range(0, len(chunks), batch_size):
                batch = np.stack(chunks[offset:offset + batch_size])
                embeddings.append(session.run(["embedding"], {"inputs": batch})[0])
            embeddings = np.concatenate(embeddings)
            cursor, vectors = 0, []
            for count in counts:
                vectors.append(embeddings[cursor:cursor + count].mean(0))
                cursor += count
            arrays[feature_name] = np.asarray(vectors, dtype="float32")
        arrays["tabular_plain"] = np.stack([row[3] for row in prepared])
        arrays["tabular_white"] = np.stack([row[4] for row in prepared])
        arrays["n_chunks"] = np.array([len(row[1]) for row in prepared], dtype=np.int64)
        for value in arrays.values():
            if not np.isfinite(value).all():
                raise ValueError(f"Non-finite features in block {start}")
        temporary = destination.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(destination)
        progress = dict(completed=min(start + block_size, len(items)), total=len(items),
                        elapsed_seconds=round(time.monotonic() - started, 1), last_block=destination.name)
        (out / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
        print(json.dumps(progress), flush=True)
    blocks = [np.load(out / f"block_{start:06d}.npz") for start in range(0, len(items), block_size)]
    joined = {key: np.concatenate([b[key] for b in blocks]) for key in blocks[0].files}
    for block in blocks:
        block.close()
    final = out / "features.npz"
    temporary = final.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **joined)
    temporary.replace(final)
    meta = {k: v for k, v in identity.items() if k != "file_ids"}
    meta.update(n_files=len(items), n_chunks=int(joined["n_chunks"].sum()),
                shapes={key: list(value.shape) for key, value in joined.items()},
                features_sha256=digest(final), complete=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    main(out=args.out, threads=args.threads, batch_size=args.batch, limit=args.limit)
