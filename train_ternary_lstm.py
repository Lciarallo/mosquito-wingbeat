"""LSTM/Bi-LSTM comparison on the frozen, PCM-disjoint ternary OOF population."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from torres_lstm import MelSequence, TorresLSTM, digest, predict_probabilities, write_json
from ternary_protocol import CLASS_NAMES, sample_weights, check_partition, calibrate_policy, apply_policy, excluded_drive_audio_mask
from ternary_reporting import evaluate_predictions

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "results/ternary_verified_2026_10_14_corrections"
DEFAULT_OUTPUT = ROOT / "results/ternary_lstm_2026_10_15"
SEED = 20261015
SETTINGS = dict(hidden=128, layers=2, dropout=.2, lr=.0006, batch_size=256,
                max_epochs=30, patience=5, gradient_clip=3., gamma=.5)


def read_json(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


class GroupPool:
    """Vectorized activity pooling; group truth matches the existing max-label rule."""
    def __init__(self, frame):
        self.groups, self.inverse = np.unique(frame.group.astype(str).to_numpy(), return_inverse=True)
        self.counts = np.bincount(self.inverse)
        self.truth = frame.groupby("group").true_label.max().reindex(self.groups).to_numpy(int)
        self.blocks = frame.groupby("group").pcm_block.first().reindex(self.groups).astype(str).to_numpy()
        self.sources = frame.groupby("group").domain.first().reindex(self.groups).astype(str).to_numpy()

    def probabilities(self, probabilities):
        p = np.asarray(probabilities, dtype=float)
        if p.shape != (len(self.inverse), 3) or not np.isfinite(p).all() or (p < 0).any():
            raise ValueError("Invalid/alignment error in ternary probabilities.")
        if not np.allclose(p.sum(1), 1, atol=1e-5):
            raise ValueError("Probabilities must sum to one.")
        weight = (1 - p[:, 0].clip(0, 1)) ** 1.5
        totals = np.bincount(self.inverse, weights=weight, minlength=len(self.groups))
        weight[totals[self.inverse] < 1e-6] = 1
        totals = np.bincount(self.inverse, weights=weight, minlength=len(self.groups))
        result = np.zeros((len(self.groups), 3))
        np.add.at(result, self.inverse, p * weight[:, None])
        result /= totals[:, None]
        return result / result.sum(1, keepdims=True)

    def table(self, probabilities):
        p = self.probabilities(probabilities)
        table = pd.DataFrame(dict(group=self.groups, true_label=self.truth, pcm_block=self.blocks,
                                  source=self.sources, persistence_available=False, persistence_q=np.nan))
        for c in range(3):
            table[f"lr_gated_{c}"] = p[:, c]
            table[f"prob_{c}"] = p[:, c]
        return table

    def loss(self, probabilities):
        p = self.probabilities(probabilities)
        count = np.bincount(self.truth, minlength=3)
        return float(np.average(-np.log(p[np.arange(len(p)), self.truth].clip(1e-12)),
                                 weights=1 / count[self.truth]))


def frontend():
    return MelSequence(sample_rate=8000, n_fft=1024, hop=128, bands=64, fmin=80, fmax=4000)


def load_inputs(reference, device):
    frame = pd.read_csv(reference / "segments.csv.gz", low_memory=False)
    if excluded_drive_audio_mask(frame).any():
        raise ValueError("Drive swarm audio is excluded from training, calibration and testing.")
    partitions = pd.read_csv(reference / "partitions.csv.gz", low_memory=False)
    if set(frame.bucket) != {"long"} or not np.allclose(frame.window_duration_s, .992):
        raise ValueError("This frozen study expects the actual 0.992-second retained windows.")
    if frame.row_id.duplicated().any() or set(frame.true_label) != {0, 1, 2}:
        raise ValueError("Invalid reference rows or ternary labels.")
    tests = partitions.loc[partitions.role.eq("test"), ["row_id", "fold"]]
    if tests.row_id.duplicated().any() or set(tests.row_id) != set(frame.row_id):
        raise ValueError("Reference test folds must cover every row once.")
    folds = frame.row_id.map(tests.set_index("row_id").fold).to_numpy(int)
    if frame.assign(outer_fold=folds).groupby("pcm_block").outer_fold.nunique().max() != 1:
        raise ValueError("PCM block crosses outer test folds.")
    wave_path = ROOT / "data/cache/reliability_2026_10_01/long_8k.npy"
    identity = dict(segments_sha256=digest(reference / "segments.csv.gz"),
        partitions_sha256=digest(reference / "partitions.csv.gz"), waveform_sha256=digest(wave_path),
        frontend=frontend().config, frontend_code_sha256=digest(ROOT / "torres_lstm.py"))
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = ROOT / "data/cache/ternary_lstm"
    cache.mkdir(parents=True, exist_ok=True)
    path, marker = cache / f"{key}.npy", cache / f"{key}.json"
    if not path.exists():
        waves = np.load(wave_path, mmap_mode="r")
        values = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(len(frame), 55, 64))
        model = frontend().to(device).eval()
        indices = frame.wave_index.to_numpy(int)
        with torch.inference_mode():
            for start in range(0, len(frame), 256):
                batch = torch.tensor(np.array(waves[indices[start:start + 256]]), device=device)
                values[start:start + len(batch)] = model(batch).cpu().numpy()
        values.flush()
        write_json(marker, dict(identity=identity, feature_sha256=digest(path)))
    metadata = read_json(marker)
    if identity != metadata["identity"] or digest(path) != metadata["feature_sha256"]:
        raise ValueError("Feature cache verification failed.")
    return frame, np.load(path, mmap_mode="r"), folds, dict(**identity, features=str(path.relative_to(ROOT)),
                                                          feature_sha256=metadata["feature_sha256"])


def normalized(raw, fit, path):
    mean = raw[fit].mean((0, 1), keepdims=True)
    std = raw[fit].std((0, 1), keepdims=True).clip(.001)
    np.savez(path, mean=mean, std=std)
    return ((raw - mean) / std).astype("float32")


def train_fit(name, frame, x, fit, validation, directory, device, seed, fixed_epochs=None):
    torch.manual_seed(seed)
    model = TorresLSTM(classes=3, features=64, hidden=SETTINGS["hidden"],
                       layers=2, dropout=.2, bidirectional=name == "bilstm").to(device)
    values = torch.tensor(np.array(x[fit]), device=device)
    target = torch.tensor(frame.true_label.to_numpy(int)[fit], device=device)
    weights = torch.tensor(sample_weights(frame.iloc[fit], target.cpu().numpy(), gamma=.5),
                           dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS["lr"])
    pool = GroupPool(frame.iloc[validation]) if validation is not None else None
    best, stale, best_epoch = float("inf"), 0, 0
    history = []
    start = time.monotonic()
    maximum = fixed_epochs if fixed_epochs is not None else SETTINGS["max_epochs"]
    for epoch in range(1, maximum + 1):
        model.train()
        order = torch.multinomial(weights, len(fit), replacement=True)
        total_loss, seen = 0., 0
        for begin in range(0, len(order), SETTINGS["batch_size"]):
            ix = order[begin:begin + SETTINGS["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(values[ix]), target[ix])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), SETTINGS["gradient_clip"])
            optimizer.step()
            total_loss += float(loss.detach()) * len(ix)
            seen += len(ix)
        row = dict(epoch=epoch, train_loss=total_loss / seen)
        if validation is not None:
            p = predict_probabilities(model, x[validation], device, batch_size=512)
            loss = pool.loss(p)
            row.update(validation_group_loss=loss, validation_accuracy=float((pool.probabilities(p).argmax(1) == pool.truth).mean()))
            if loss < best - 1e-4:
                best, stale, best_epoch = loss, 0, epoch
                state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                torch.save(dict(config=model.config, classes=CLASS_NAMES, state_dict=state), directory / "inner_best.pt")
            else:
                stale += 1
        history.append(row)
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps(dict(context=directory.parent.name, model=name, phase="refit" if fixed_epochs else "selection",
                                  **row, seconds=time.monotonic() - start)), flush=True)
        if fixed_epochs is None and stale >= SETTINGS["patience"]:
            break
    if validation is not None:
        model.load_state_dict(torch.load(directory / "inner_best.pt", map_location="cpu", weights_only=True)["state_dict"])
    summary = dict(best_epoch=best_epoch if validation is not None else maximum,
                   validation_loss=best if validation is not None else None, epochs_run=len(history),
                   parameters=sum(p.numel() for p in model.parameters()), seconds=time.monotonic() - start)
    write_json(directory / ("inner_history.json" if validation is not None else "refit_history.json"), history)
    return model, summary


def run_context(name, context, frame, raw, fit, device, directory, seed):
    directory.mkdir(parents=True)
    splitter = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
    it, iv = next(splitter.split(np.zeros(len(fit)), frame.true_label.to_numpy()[fit], frame.pcm_block.iloc[fit]))
    train, validation = fit[it], fit[iv]
    check_partition(frame, train, validation)
    table = frame.iloc[np.r_[train, validation]][["row_id", "group", "path", "pcm_block", "true_label"]].copy()
    table["role"] = ["fit"] * len(train) + ["validation"] * len(validation)
    table.to_csv(directory / "inner_partitions.csv.gz", index=False)
    x = normalized(raw, train, directory / "inner_scaler.npz")
    model, selection = train_fit(name, frame, x, train, validation, directory, device, seed)
    p = predict_probabilities(model, x[validation], device, batch_size=512)
    groups = GroupPool(frame.iloc[validation]).table(p)
    policy = calibrate_policy(groups, "lr", temporal=False)
    policy["selection_scope"] = "inner_holdout_within_outer_training"
    groups.to_csv(directory / "validation_predictions.csv", index=False)
    write_json(directory / "selection.json", dict(**selection, policy=policy,
        inner_fit_rows=frame.row_id.iloc[train].tolist(), validation_rows=frame.row_id.iloc[validation].tolist()))
    del model, x
    # A new scaler and a new model use all outer training rows, never outer test.
    x = normalized(raw, fit, directory / "scaler.npz")
    model, fitted = train_fit(name, frame, x, fit, None, directory, device, seed,
                              fixed_epochs=selection["best_epoch"])
    path = directory / "model.pt"
    torch.save(dict(config=model.config, classes=CLASS_NAMES, state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    frontend=frontend().config, policy=policy, sample_rate=8000), path)
    write_json(directory / "freeze.json", dict(context=context, model=name, frozen_utc=now(),
        frozen_before_outer_test=True, fit_rows=frame.row_id.iloc[fit].tolist(),
        checkpoint_sha256=digest(path), scaler_sha256=digest(directory / "scaler.npz"),
        selection_sha256=digest(directory / "selection.json"), best_epoch=selection["best_epoch"],
        validation_loss=selection["validation_loss"], parameters=fitted["parameters"], policy=policy))
    return model, x, dict(**selection, policy=policy)


def run(output, reference, device):
    if output.exists():
        raise FileExistsError("Existing experiments are preserved: " + str(output))
    output.mkdir(parents=True)
    (output / "source").mkdir()
    sources = ["train_ternary_lstm.py", "torres_lstm.py", "ternary_protocol.py", "ternary_reporting.py"]
    for name in sources:
        shutil.copy2(ROOT / name, output / "source" / name)
    frame, raw, folds, metadata = load_inputs(reference, device)
    frame["outer_fold"] = folds
    frame.to_csv(output / "segments.csv.gz", index=False)
    shutil.copy2(reference / "partitions.csv.gz", output / "reference_partitions.csv.gz")
    write_json(output / "protocol.json", dict(created_utc=now(), settings=SETTINGS, seed=SEED,
        classes=CLASS_NAMES, inputs=metadata, reference=str(reference.relative_to(ROOT)),
        source_sha256={name:digest(ROOT / name) for name in sources}, device=str(device), torch_version=str(torch.__version__),
        outer_folds=5, inner_holdout_folds=4, group_truth="maximum window label as frozen reference",
        group_pooling="activity weights (1-p0)^1.5; exact low-weight fallback as reference",
        test_already_known_retrospective=True, pretrained_infection_weights_used=False,
        windows=len(frame), groups=int(frame.group.nunique()), pcm_blocks=int(frame.pcm_block.nunique()),
        normalization_fitted_on_training_only=True, primary_oof_inference="deterministic; calibration from inner holdout only",
        calibration_fpr=dict(noise=.0075, other=.0175), final_mc_inference_passes=100))
    oof = {name:np.full((len(frame), 3), np.nan, dtype="float32") for name in ["lstm", "bilstm"]}
    policies, freezes = {}, {}
    for fold in range(5):
        fit, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        check_partition(frame, fit, test)
        for name in oof:
            path = output / f"outer_{fold}" / name
            model, x, selected = run_context(name, f"outer_{fold}", frame, raw, fit, device, path, SEED + fold * 101)
            freezes[f"outer_{fold}/{name}"] = digest(path / "freeze.json")
            policies[f"{fold}/{name}"] = selected["policy"]
            oof[name][test] = predict_probabilities(model, x[test], device, batch_size=512)
            del model, x
    final_candidates = []
    for name in oof:
        path = output / "final" / name
        model, x, selected = run_context(name, "final", frame, raw, np.arange(len(frame)), device, path, SEED + 1000)
        freezes[f"final/{name}"] = digest(path / "freeze.json")
        final_candidates.append(dict(model=name, validation_loss=selected["validation_loss"], best_epoch=selected["best_epoch"]))
        del model, x
    write_json(output / "freeze.json", dict(frozen_utc=now(), context_freeze_sha256=freezes,
        selected_final_neural_model=min(final_candidates, key=lambda c:c["validation_loss"])["model"],
        final_candidates=final_candidates, selection_uses_outer_test=False))
    reports = []
    for name, p in oof.items():
        assert np.isfinite(p).all()
        np.save(output / f"{name}_window_oof.npy", p)
        groups = GroupPool(frame).table(p)
        group_fold = frame.groupby("group").outer_fold.first()
        groups["fold"] = groups.group.map(group_fold)
        groups["argmax_prediction"] = groups[[f"prob_{c}" for c in range(3)]].to_numpy().argmax(1)
        groups["calibrated_prediction"] = -1
        for fold in range(5):
            mask = groups.fold.eq(fold)
            groups.loc[mask, "calibrated_prediction"] = apply_policy(groups.loc[mask], policies[f"{fold}/{name}"])
        assert not (groups.calibrated_prediction < 0).any()
        groups.to_csv(output / f"{name}_event_oof.csv", index=False)
        for decision in ["argmax", "calibrated"]:
            metric = evaluate_predictions(groups.true_label, groups[f"{decision}_prediction"], sources=groups.source)
            reports.append(dict(model=name, decision=decision, **metric))
    base = read_json(reference / "event_level_metrics.json")["models"]
    if isinstance(base, dict):
        baseline = base["activity_gated_log_odds_cascade_event"]
    else:
        baseline = next(r for r in base if r["model"] == "activity_gated_log_odds_cascade_event")
    write_json(output / "metrics.json", dict(reference=dict(model="cascata LR + HistGB", **baseline), neural_models=reports))
    write_json(output / "complete.json", dict(complete=True, models=2, outer_folds=5, group_oof_coverage=1494,
        windows=len(frame), finished_utc=now()))
    for row in reports:
        print(json.dumps({k:row[k] for k in ["model", "decision", "accuracy", "aedes_recall", "noise_to_aedes_fpr", "other_to_aedes_fpr"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
    run(args.output.resolve(), args.reference.resolve(), device)
