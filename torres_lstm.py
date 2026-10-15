"""LSTM/Bi-LSTM and selective Monte Carlo inference after Torres et al. (2025).

The explicit dropout between single-layer LSTMs implements inter-layer dropout
without the MIOpen recurrent-dropout compiler dependency. It also makes stochastic
inference independent of batch-normalization or other training-mode behavior.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


AUTHOR_REPOSITORY = "https://github.com/NakanoMariko/Dengue-Infected-Mosquito-Detection"
AUTHOR_COMMIT = "6c96ccd5347cf5bafc9b424579700af7b5ffb7ce"
SEED = 20261015


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class TorresLSTM(nn.Module):
    """Time-major features in a batch: (windows, time, 32 mel bands).

    Readout follows the public author implementation: linear(last output).
    Bidirectional last-output readout is kept explicit; it is not silently
    replaced by concatenated final hidden states.
    """

    def __init__(self, classes=2, hidden=256, layers=2, dropout=0.2, bidirectional=False, features=32):
        super().__init__()
        if classes < 2 or hidden < 1 or layers < 1 or features < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid recurrent architecture.")
        self.config = dict(classes=classes, hidden=hidden, layers=layers,
                           dropout=dropout, bidirectional=bidirectional, features=features)
        directions = 2 if bidirectional else 1
        self.recurrent = nn.ModuleList([
            nn.LSTM(features if layer == 0 else hidden * directions, hidden,
                    num_layers=1, batch_first=True, bidirectional=bidirectional)
            for layer in range(layers)
        ])
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(layers - 1)])
        self.linear = nn.Linear(hidden * directions, classes)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.config["features"] or x.shape[1] == 0:
            raise ValueError(f"Expected nonempty (batch, time, {self.config['features']}) feature sequences.")
        for layer, recurrent in enumerate(self.recurrent):
            x, _ = recurrent(x)
            if layer < len(self.dropout):
                x = self.dropout[layer](x)
        return self.linear(x[:, -1, :])

    def load_author_state(self, state):
        """Translate public nn.LSTM state names without loading executable code."""
        own = self.state_dict()
        translated = {}
        consumed = set()
        for key in own:
            if key.startswith("recurrent."):
                _, layer, parameter = key.split(".", 2)
                source = "lstm." + parameter.replace("_l0", "_l" + layer)
            else:
                source = key
            translated[key] = state[source]
            consumed.add(source)
        if consumed != set(state):
            raise ValueError("Unexpected author checkpoint keys.")
        self.load_state_dict(translated, strict=True)


@contextmanager
def stochastic_dropout(model):
    """Activate only explicit dropout; restore every module's previous mode."""
    previous = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        yield
    finally:
        for module, training in previous:
            module.training = training


def predict_probabilities(model, x, device="cpu", batch_size=128):
    if not len(x):
        raise ValueError("Cannot predict an empty dataset.")
    previous = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.inference_mode():
            values = [model(torch.as_tensor(np.array(x[start:start + batch_size]),
                                            dtype=torch.float32, device=device)).softmax(-1).cpu().numpy()
                      for start in range(0, len(x), batch_size)]
        return np.concatenate(values)
    finally:
        for module, training in previous:
            module.training = training


def monte_carlo_probabilities(model, x, device="cpu", passes=100,
                              batch_size=64, draw_batch=10, seed=SEED):
    """Returns (passes, examples, classes); independent dropout per example/draw."""
    if passes < 2 or batch_size < 1 or draw_batch < 1 or not len(x):
        raise ValueError("MC inference needs >=2 passes and nonempty input.")
    device = torch.device(device)
    devices = [device.index or 0] if device.type == "cuda" else []
    output = np.empty((passes, len(x), model.config["classes"]), dtype="float32")
    with torch.random.fork_rng(devices=devices), stochastic_dropout(model), torch.inference_mode():
        torch.manual_seed(seed)
        for start in range(0, len(x), batch_size):
            value = torch.as_tensor(np.array(x[start:start + batch_size]),
                                    dtype=torch.float32, device=device)
            for draw in range(0, passes, draw_batch):
                count = min(draw_batch, passes - draw)
                probs = model(value.repeat(count, 1, 1)).softmax(-1)
                output[draw:draw + count, start:start + len(value)] = (
                    probs.reshape(count, len(value), -1).cpu().numpy())
    if not np.isfinite(output).all() or not np.allclose(output.sum(-1), 1, atol=1e-5):
        raise ValueError("Invalid stochastic probabilities.")
    return output


def aggregate_groups(frame, probabilities):
    """Average windows per original recording, preserving each MC draw."""
    required = {"group", "target"}
    if not required.issubset(frame):
        raise ValueError("Missing evaluation identities.")
    frame = frame.assign(group=frame.group.astype(str))
    if frame.groupby("group").target.nunique().max() != 1:
        raise ValueError("Every evaluation group must have one target.")
    value = np.asarray(probabilities)
    if value.ndim not in (2, 3) or value.shape[-2] != len(frame):
        raise ValueError("Probabilities do not align with metadata.")
    groups, inverse = np.unique(frame.group.astype(str).to_numpy(), return_inverse=True)
    counts = np.bincount(inverse)
    batched = value[None] if value.ndim == 2 else value
    out = np.zeros((len(batched), len(groups), value.shape[-1]), dtype="float64")
    for draw in range(len(batched)):
        np.add.at(out[draw], inverse, batched[draw])
    out /= counts[None, :, None]
    truth = frame.groupby("group").target.first().reindex(groups).to_numpy(dtype=int)
    return groups, truth, out[0] if value.ndim == 2 else out


def vote_statistics(probabilities):
    value = np.asarray(probabilities, dtype="float64")
    if value.ndim != 3 or len(value) < 2 or not np.isfinite(value).all():
        raise ValueError("Expected finite MC draws by example by class.")
    if (value < 0).any() or not np.allclose(value.sum(-1), 1, atol=1e-5):
        raise ValueError("MC draws must be normalized probabilities.")
    decisions = value.argmax(-1)
    counts = np.stack([(decisions == c).sum(0) for c in range(value.shape[-1])], axis=1)
    mean = value.mean(0)
    entropy = lambda p: -(p * np.log(p.clip(1e-12))).sum(-1)
    return dict(prediction=counts.argmax(-1), mean_prediction=mean.argmax(-1),
                mean_probability=mean, vote_counts=counts,
                uncertainty=1 - counts.max(-1) / len(value),
                predictive_entropy=entropy(mean),
                mutual_information=np.maximum(0, entropy(mean) - entropy(value).mean(0)))


def classification_metrics(y, prediction, classes):
    y, prediction = np.asarray(y, dtype=int), np.asarray(prediction, dtype=int)
    labels = np.arange(len(classes))
    if len(y) == 0:
        return dict(total=0, correct=0, accuracy=None, balanced_accuracy=None,
                    macro_f1=None, confusion_matrix=np.zeros((len(classes), len(classes)), int).tolist())
    return dict(total=len(y), correct=int((y == prediction).sum()),
                accuracy=float(accuracy_score(y, prediction)),
                balanced_accuracy=float(balanced_accuracy_score(y, prediction)),
                macro_f1=float(f1_score(y, prediction, labels=labels, average="macro", zero_division=0)),
                confusion_matrix=confusion_matrix(y, prediction, labels=labels).tolist())


def selective_metrics(y, prediction, uncertainty, classes, threshold=0):
    y, prediction, uncertainty = map(np.asarray, (y, prediction, uncertainty))
    if not 0 <= threshold < 1 or not (len(y) == len(prediction) == len(uncertainty)):
        raise ValueError("Invalid selective evaluation inputs.")
    accept = uncertainty <= threshold + 1e-12
    result = classification_metrics(y[accept], prediction[accept], classes)
    emitted = np.where(accept, prediction, len(classes))
    matrix = np.zeros((len(classes), len(classes) + 1), dtype=int)
    np.add.at(matrix, (y.astype(int), emitted.astype(int)), 1)
    result.update(threshold=float(threshold), input_total=len(y), accepted=int(accept.sum()),
                  abstained=int((~accept).sum()), coverage=float(accept.mean()) if len(y) else 0.0,
                  overall_correct_decision_rate=float(((y == prediction) & accept).mean()) if len(y) else 0.0,
                  confusion_with_abstention=matrix.tolist(),
                  per_class_coverage={name: float(accept[y == i].mean()) if (y == i).any() else None
                                      for i, name in enumerate(classes)})
    return result


def purge_partition_collisions(frame):
    """Keep a PCM component only in its highest-priority historical partition."""
    required = {"group", "pcm_block", "split", "content_sha256"}
    if not required.issubset(frame) or frame[list(required)].isna().any().any():
        raise ValueError("Incomplete identities for partition auditing.")
    rank = {"train": 0, "calibration": 1, "validation": 2, "test": 3}
    if not set(frame.split).issubset(rank):
        raise ValueError("Unknown partition.")
    work = frame.copy()
    priority = work.split.map(rank)
    maximum = priority.groupby(work.pcm_block).transform("max")
    kept, removed = work.loc[priority == maximum].copy(), work.loc[priority != maximum].copy()
    for column in ["group", "pcm_block", "content_sha256"]:
        if kept.groupby(column).split.nunique().max() > 1:
            raise ValueError("Cross-partition identity remains: " + column)
    return kept, removed


class MelSequence(nn.Module):
    """Documented project adaptation: 16 kHz, 1024 Hann, hop 256, 32 mel bands."""
    def __init__(self, sample_rate=16000, n_fft=1024, hop=256, bands=32, fmin=80, fmax=4000):
        super().__init__()
        self.config = dict(sample_rate=sample_rate, n_fft=n_fft, hop=hop,
                           bands=bands, fmin=fmin, fmax=fmax, center=False,
                           scaling="per-window zero mean/RMS; HTK mel; peak-relative dB clipped [-80,0]")
        hz = np.fft.rfftfreq(n_fft, 1 / sample_rate)
        edges = 700 * (10 ** (np.linspace(2595 * np.log10(1 + fmin / 700),
                                            2595 * np.log10(1 + fmax / 700), bands + 2) / 2595) - 1)
        bank = np.zeros((bands, len(hz)), dtype="float32")
        for i in range(bands):
            bank[i] = np.maximum(0, np.minimum((hz - edges[i]) / (edges[i + 1] - edges[i]),
                                               (edges[i + 2] - hz) / (edges[i + 2] - edges[i + 1])))
            bank[i] /= max(bank[i].sum(), 1e-12)
        self.register_buffer("bank", torch.tensor(bank))
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, wave):
        wave = wave.float() - wave.float().mean(-1, keepdim=True)
        wave = wave / wave.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        stft = torch.stft(wave, self.config["n_fft"], self.config["hop"],
                          window=self.window, center=False, return_complex=True)
        db = 10 * torch.log10((self.bank @ stft.abs().square()).clamp_min(1e-12))
        db = (db - db.amax((-2, -1), keepdim=True)).clamp(-80, 0)
        return db.transpose(-1, -2).contiguous()
