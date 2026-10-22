"""v4 amendment A1: light fine-tuning of Perch v2 on GPU (reanalysis/protocol_v4_amendment_A1.json).

The Perch ONNX graph is cut at its 'spectrogram' tensor. The spectral frontend stays frozen (ONNX, CPU);
the spectrogram->embedding backbone is converted to PyTorch with onnx2torch and checked for numerical
parity with the original ONNX. Only the last MBConv stage and the 384->1536 head convolution are trained
(convolutions 71-77 in graph order), with a presence head and an Aedes-among-mosquitoes head, on the
fitting rows of each fold only. The fine-tuned 1536-d file embeddings then enter the unchanged v4
two-stage pipeline as candidate D1_perch_ft (logistic heads, calibration, threshold, gate, few-shot).

Commands (run with .venv-v4-rocm): prepare | folds | final
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ONNX = ROOT / "data/models/perch_v2_onnx/perch_v2.onnx"
MODELS = ROOT / "data/models/perch_v2_torch"
SPEC = ROOT / "data/cache/v4_perch_spec"
OUT = ROOT / "results/v4/finetune"
SEED = 20261103
TRAINABLE_FROM = 71
TRAINABLE_SHAPES = [(232, 1392, 1), (1392, 1392, 1392), (1392, 384, 1), (384, 2304, 1), (2304, 2304, 2304),
                    (2304, 384, 1), (384, 1536, 1)]
EPOCHS, BATCH, LR_BACKBONE, LR_HEADS, WEIGHT_DECAY = 3, 32, 1e-4, 1e-3, 1e-3


def digest(path):
    import hashlib
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=float) + "\n")


# ----------------------------------------------------------------------------- model
def split_onnx(out_dir=MODELS):
    import onnx
    from onnx import utils
    out_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(ONNX)
    del model.graph.value_info[:]  # stale shape annotations break shape inference; values are unchanged
    clean = out_dir / "perch_v2_clean.onnx"
    onnx.save(model, clean)
    utils.extract_model(str(clean), str(out_dir / "frontend.onnx"), ["inputs"], ["spectrogram"])
    utils.extract_model(str(clean), str(out_dir / "backbone.onnx"), ["spectrogram"], ["embedding"])
    return out_dir / "frontend.onnx", out_dir / "backbone.onnx"


def load_backbone():
    import torch
    return torch.load(MODELS / "backbone.pt", weights_only=False)


def trainable_convolutions(backbone):
    """Pre-registered rule: convolutions from index 71 (graph order) = last MBConv stage + head conv."""
    import torch
    convs = [m for _, m in backbone.named_modules() if isinstance(m, torch.nn.Conv2d)]
    chosen = convs[TRAINABLE_FROM:]
    shapes = [(m.in_channels, m.out_channels, m.groups) for m in chosen]
    if len(convs) != 78 or shapes != TRAINABLE_SHAPES:
        raise ValueError(f"Unexpected Perch structure: {len(convs)} convs, trainable {shapes}")
    return chosen


class Tuned:
    """Backbone + presence head + Aedes-among-mosquitoes head."""

    def __init__(self, backbone):
        import torch
        self.backbone = backbone
        self.presence = torch.nn.Linear(1536, 1)
        self.aedes = torch.nn.Linear(1536, 1)

    def to(self, device):
        for m in (self.backbone, self.presence, self.aedes):
            m.to(device)
        return self

    def parameters_to_train(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        convs = trainable_convolutions(self.backbone)
        backbone = [p for m in convs for p in m.parameters()]
        for p in backbone:
            p.requires_grad_(True)
        return backbone, list(self.presence.parameters()) + list(self.aedes.parameters())


# ----------------------------------------------------------------------------- spectrogram cache
def development_frame():
    from reanalysis import v4_train as v4
    return v4.load_development()[0]


def prepare(threads=8):
    """Backbone conversion + parity check; frozen-frontend spectrograms of every development chunk (fp16)."""
    import onnxruntime as ort
    import torch
    from onnx2torch import convert
    import onnx
    from reanalysis.v3_features import prepare_audio
    frontend_path, backbone_path = split_onnx()
    backbone = convert(onnx.load(backbone_path))
    torch.save(backbone, MODELS / "backbone.pt")
    trainable_convolutions(backbone)
    frame = development_frame()
    items = list(zip(frame.file_id, frame.path))
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
    front = ort.InferenceSession(str(frontend_path), options, providers=["CPUExecutionProvider"])
    full = ort.InferenceSession(str(ONNX), options, providers=["CPUExecutionProvider"])
    SPEC.mkdir(parents=True, exist_ok=True)
    counts, specs, parity = [], [], None
    started = time.monotonic()
    for start in range(0, len(items), 64):
        with ThreadPoolExecutor(4) as executor:
            prepared = list(executor.map(prepare_audio, items[start:start + 64]))
        for _, _, white, _, _ in prepared:  # whitened chunks = the input of perch_white
            batch = np.stack(white)
            spec = front.run(["spectrogram"], {"inputs": batch})[0]
            if parity is None:
                ref = full.run(["embedding"], {"inputs": batch})[0]
                with torch.no_grad():
                    got32 = backbone.eval()(torch.from_numpy(spec)).numpy()
                    got16 = backbone(torch.from_numpy(spec.astype("float16").astype("float32"))).numpy()
                parity = dict(max_abs_diff_fp32=float(np.abs(got32 - ref).max()),
                              max_abs_diff_fp16_spectrogram=float(np.abs(got16 - ref).max()),
                              min_cosine_fp16=float(((got16 * ref).sum(1) / np.linalg.norm(got16, axis=1)
                                                     / np.linalg.norm(ref, axis=1)).min()))
                if parity["min_cosine_fp16"] < .999:
                    raise ValueError(f"Converted backbone does not match ONNX: {parity}")
            specs.append(spec.astype("float16"))
            counts.append(len(white))
        print(json.dumps(dict(completed=min(start + 64, len(items)), total=len(items),
                              elapsed_seconds=round(time.monotonic() - started, 1))), flush=True)
    joined = np.concatenate(specs)
    np.save(SPEC / "spectrograms.npy", joined)
    np.save(SPEC / "file_ids.npy", frame.file_id.to_numpy(np.int64))
    np.save(SPEC / "n_chunks.npy", np.array(counts, dtype=np.int64))
    meta = dict(onnx_sha256=digest(ONNX), backbone_onnx_sha256=digest(backbone_path),
                backbone_pt_sha256=digest(MODELS / "backbone.pt"), frontend_sha256=digest(frontend_path),
                code_sha256=digest(__file__), parity=parity, n_files=len(items), n_chunks=int(joined.shape[0]),
                spectrograms_sha256=digest(SPEC / "spectrograms.npy"), complete=True)
    write_json(SPEC / "meta.json", meta)
    print(json.dumps(meta, indent=2), flush=True)


def load_spec(frame):
    meta = json.loads((SPEC / "meta.json").read_text())
    if not meta.get("complete") or digest(SPEC / "spectrograms.npy") != meta["spectrograms_sha256"]:
        raise ValueError("Incomplete or changed spectrogram cache")
    ids = np.load(SPEC / "file_ids.npy")
    if not np.array_equal(ids, frame.file_id.to_numpy(np.int64)):
        raise ValueError("Spectrogram cache order differs from the development frame")
    counts = np.load(SPEC / "n_chunks.npy")
    return np.load(SPEC / "spectrograms.npy", mmap_mode="r"), np.r_[0, np.cumsum(counts)], meta


# ----------------------------------------------------------------------------- training
def chunk_weights(target, counts, fit):
    """Each fitting file carries one unit of mass split over its chunks; classes balanced per task."""
    from reanalysis import v3_train as v3
    y = target[fit]
    presence_w = v3.balanced_weights((y > 0).astype(int))
    mosquito = y > 0
    aedes_w = np.zeros(len(fit))
    aedes_w[mosquito] = v3.balanced_weights((y[mosquito] == 2).astype(int))
    rows = np.repeat(np.arange(len(fit)), counts[fit])
    split = 1. / np.repeat(counts[fit], counts[fit])
    return rows, presence_w[rows] * split, aedes_w[rows] * split


def train(frame, fit, seed, device="cuda"):
    import torch
    spec, offsets, _ = load_spec(frame)
    counts = np.diff(offsets)
    target = frame.target.to_numpy(int)
    rows, w_presence, w_aedes = chunk_weights(target, counts, fit)
    chunk_index = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in fit])
    y_presence = (target[fit][rows] > 0).astype("float32")
    y_aedes = (target[fit][rows] == 2).astype("float32")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Tuned(load_backbone()).to(device)
    backbone_params, head_params = model.parameters_to_train()
    opt = torch.optim.AdamW([dict(params=backbone_params, lr=LR_BACKBONE), dict(params=head_params, lr=LR_HEADS)],
                            weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(seed)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    history = []
    for epoch in range(EPOCHS):
        model.backbone.train()
        order = rng.permutation(len(chunk_index))
        total, n = 0., 0
        for start in range(0, len(order), BATCH):
            ix = order[start:start + BATCH]
            x = torch.from_numpy(spec[chunk_index[ix]].astype("float32")).to(device)
            emb = model.backbone(x)
            lp, la = model.presence(emb).squeeze(1), model.aedes(emb).squeeze(1)
            t = lambda a: torch.from_numpy(a[ix]).to(device)
            loss = (bce(lp, t(y_presence), weight=t(w_presence.astype("float32")), reduction="sum")
                    + bce(la, t(y_aedes), weight=t(w_aedes.astype("float32")), reduction="sum")) / len(ix)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, n = total + float(loss) * len(ix), n + len(ix)
        history.append(dict(epoch=epoch, mean_loss=total / n))
        print(json.dumps(history[-1]), flush=True)
    return model, history


def embed(model, frame, device="cuda"):
    import torch
    spec, offsets, _ = load_spec(frame)
    model.backbone.eval()
    out = np.zeros((len(frame), 1536), dtype="float32")
    with torch.no_grad():
        for start in range(0, len(frame), 64):
            files = range(start, min(start + 64, len(frame)))
            lo, hi = offsets[files.start], offsets[files.stop]
            emb = model.backbone(torch.from_numpy(spec[lo:hi].astype("float32")).to(device)).cpu().numpy()
            for i in files:
                out[i] = emb[offsets[i] - lo:offsets[i + 1] - lo].mean(0)
    return out


def run(tag, frame, fit, seed):
    import torch
    directory = OUT / tag
    if (directory / "embeddings.json").exists():
        return
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    model, history = train(frame, fit, seed)
    torch.save(dict(backbone=[p.detach().cpu() for p in model.parameters_to_train()[0]],
                    presence=model.presence.state_dict(), aedes=model.aedes.state_dict()), directory / "model.pt")
    emb = embed(model, frame)
    if not np.isfinite(emb).all():
        raise ValueError(f"Non-finite fine-tuned embeddings in {tag}")
    np.save(directory / "embeddings.npy", emb)
    write_json(directory / "embeddings.json", dict(
        tag=tag, seed=seed, training_file_ids=frame.file_id.iloc[fit].tolist(), history=history,
        epochs=EPOCHS, batch=BATCH, lr_backbone=LR_BACKBONE, lr_heads=LR_HEADS, weight_decay=WEIGHT_DECAY,
        trainable_from_conv=TRAINABLE_FROM, model_sha256=digest(directory / "model.pt"),
        embeddings_sha256=digest(directory / "embeddings.npy"), spectrogram_meta_sha256=digest(SPEC / "meta.json"),
        device=torch.cuda.get_device_name(0), torch=torch.__version__, hip=torch.version.hip,
        seconds=round(time.monotonic() - started, 1),
        note="GPU kernels are not bitwise deterministic; seeds fix data order and initialisation."))


def load_tuned(tag, device="cuda"):
    """Rebuild a fine-tuned model (frozen backbone + saved trainable convolutions + heads) for new audio."""
    import torch
    directory = OUT / tag
    meta = json.loads((directory / "embeddings.json").read_text())
    if digest(directory / "model.pt") != meta["model_sha256"]:
        raise ValueError(f"Changed fine-tuned model: {tag}")
    saved = torch.load(directory / "model.pt", weights_only=False)
    model = Tuned(load_backbone())
    params, _ = model.parameters_to_train()
    with torch.no_grad():
        for p, value in zip(params, saved["backbone"], strict=True):
            p.copy_(value)
    model.presence.load_state_dict(saved["presence"])
    model.aedes.load_state_dict(saved["aedes"])
    return model.to(device)


def folds():
    from reanalysis import v3_train as v3
    from reanalysis import v4_train as v4
    v4.check_freeze()
    frame = development_frame()
    for k, part in enumerate(v4.partitions(frame)):
        fit, _ = v3.split_fit_calibration(frame, part["train"], v4.SEED + k)
        run(part["tag"], frame, fit, SEED + k)


def final():
    from reanalysis import v3_train as v3
    from reanalysis import v4_train as v4
    v4.check_freeze()
    frame = development_frame()
    fit, _ = v3.split_fit_calibration(frame, np.arange(len(frame)), v4.SEED + 99)
    run("final", frame, fit, SEED + 99)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "folds", "final"])
    stage = parser.parse_args().stage
    {"prepare": prepare, "folds": folds, "final": final}[stage]()
