"""Detector v3 - frozen Perch 2.0 embeddings (ONNX port of the official weights).

Audio path: decode -> 8 kHz (same as v2) -> 100-3400 Hz band-pass -> optional
per-recording background whitening -> 32 kHz -> 5 s chunks (max 6 per file, evenly
spaced; shorter files centred and zero-padded) -> unit RMS -> Perch embedding (1536).
Upsampling to 32 kHz adds no information above 4 kHz.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.signal import istft, resample_poly, sosfiltfilt, stft

from reanalysis import frontend as fe

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "data/models/perch_v2_onnx/perch_v2.onnx"
CHUNK = 160000
MAX_CHUNKS = 6


def whiten(x, sr=8000):
    """Divide the STFT magnitude by the recording's own per-band noise floor (20th percentile)."""
    if len(x) < 1024:
        return x
    f, t, Z = stft(x, sr, nperseg=512, noverlap=384)
    floor = np.percentile(np.abs(Z), 20, axis=1, keepdims=True) + 1e-8
    _, y = istft(Z / floor, sr, nperseg=512, noverlap=384)
    return y[: len(x)].astype("float32")


def chunks(path, whitened):
    x = fe.decode(ROOT / path)
    if len(x) > 30:
        x = sosfiltfilt(fe.SOS, x).astype("float32")
    if whitened:
        x = whiten(x)
    y = resample_poly(x, 4, 1).astype("float32")  # 8 -> 32 kHz
    if len(y) <= CHUNK:
        pad = CHUNK - len(y); y = np.pad(y, (pad // 2, pad - pad // 2)); starts = [0]
    else:
        n = len(y) // CHUNK
        starts = (np.unique(np.linspace(0, n - 1, min(n, MAX_CHUNKS)).round().astype(int)) * CHUNK).tolist()
    out = []
    for s in starts:
        c = y[s:s + CHUNK]
        r = float(np.sqrt(np.mean(c ** 2)))
        out.append((c / max(r, 1e-8) * 0.05).astype("float32"))
    return out, [s / 32000 for s in starts]


def main(manifest, out_prefix, whitened=False, threads=24, batch=32):
    import onnxruntime as ort
    from concurrent.futures import ThreadPoolExecutor
    so = ort.SessionOptions(); so.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(MODEL), so, providers=["CPUExecutionProvider"])
    man = pd.read_csv(manifest)
    rows, embs, buf, meta = [], [], [], []
    def flush():
        if buf:
            embs.append(sess.run(["embedding"], {"inputs": np.stack(buf)})[0].astype("float32")); buf.clear()
    with ThreadPoolExecutor(8) as ex:
        for n, (i, (cs, st)) in enumerate(zip(man.index, ex.map(lambda p: chunks(p, whitened), man.path)), 1):
            for c, s in zip(cs, st):
                buf.append(c); rows.append(dict(file_index=int(i), start_s=s))
                if len(buf) == batch: flush()
            if n % 1000 == 0: print("files", n, "/", len(man), flush=True)
    flush()
    E = np.concatenate(embs)
    assert len(E) == len(rows) and np.isfinite(E).all()
    np.save(f"{out_prefix}.npy", E); pd.DataFrame(rows).to_csv(f"{out_prefix}_chunks.csv", index=False)
    meta = dict(model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(), whitened=whitened,
                n_chunks=len(rows), dim=int(E.shape[1]),
                manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
                embed_sha256=hashlib.sha256(Path(f"{out_prefix}.npy").read_bytes()).hexdigest())
    Path(f"{out_prefix}_meta.json").write_text(json.dumps(meta, indent=1))
    print(meta)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], whitened=len(sys.argv) > 3 and sys.argv[3] == "whiten")
