"""Run Torres-style recurrent/MC experiments without overwriting earlier models.

Tasks remain separate: DENV-2 infection on author mel tensors; species recognition
on the project's Dryad recordings. Aedes species labels never become infection
labels. All tuning precedes test inference and selective coverage is reported.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from torres_lstm import (AUTHOR_COMMIT, AUTHOR_REPOSITORY, SEED, TorresLSTM, MelSequence,
    aggregate_groups, classification_metrics, digest, monte_carlo_probabilities,
    predict_probabilities, purge_partition_collisions, selective_metrics, vote_statistics, write_json)

ROOT = Path(__file__).resolve().parent
AUTHOR = ROOT / "data/external/torres_2025"
DEFAULT_OUTPUT = ROOT / "results/torres_2025_2026_10_15"


def log(**row):
    print(json.dumps(row, allow_nan=False), flush=True)


def load_infection():
    path = AUTHOR / "Dataset/Melspec.json"
    manifest = json.loads((AUTHOR / "manifest.json").read_text())
    expected = manifest.get("expanded_json_sha256")
    if not expected or digest(path) != expected or manifest["commit"] != AUTHOR_COMMIT:
        raise ValueError("Run fetch_torres_2025.py to verify the pinned dataset.")
    data = json.loads(path.read_text())
    x, y = np.asarray(data["mspec"], dtype="float32"), np.asarray(data["labels"], dtype=int)
    if x.shape != (300, 188, 32) or not np.isfinite(x).all() or set(y) != {0, 1}:
        raise ValueError("Unexpected public dataset shape/labels.")
    keys = [hashlib.sha256(row.tobytes()).hexdigest() for row in x]
    frame = pd.DataFrame(dict(row_id=np.arange(len(x)), group=keys, pcm_block=keys,
                              target=y, label=np.asarray(["infected_denv2", "healthy"])[y],
                              content_sha256=keys, split="train"))
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    development, test = next(outer.split(x, y, frame.group))
    inner = StratifiedGroupKFold(5, shuffle=True, random_state=SEED + 1)
    train, validation = next(inner.split(x[development], y[development], frame.group.iloc[development]))
    frame.loc[development[validation], "split"] = "validation"
    frame.loc[test, "split"] = "test"
    if frame.groupby("group").split.nunique().max() != 1:
        raise ValueError("Duplicate tensor crosses partitions.")
    metadata = dict(task="DENV-2 infection status", classes=["infected_denv2", "healthy"],
        input_shape=list(x.shape), source_json_sha256=digest(path), author_commit=AUTHOR_COMMIT,
        identity_scope="exact feature-tensor hash; individual/recording identifiers absent from public JSON",
        unknown_individual_overlap=True, original_raw_audio_available_for_all_examples=False,
        feature_scaling="original author mel tensors, no fitted scaling", article_sample_rate=48000,
        article_fft=1024, article_hop=256, article_mel_bands=32,
        label_source="authors Testing_LSTM_MC_DO.py class order: Infected=0, Healthy=1")
    return frame, x, metadata, pd.DataFrame()


def load_species(device):
    source = pd.read_csv(ROOT / "results/expanded/segments.csv", low_memory=False)
    historical = pd.read_csv(ROOT / "results/ifsc_melhorias_2026_10_11/partitions.csv")
    frame = source.loc[source.domain.eq("dryad") & source.label.ne("noise") & source.quality_ok].copy()
    if len(frame) != len(historical) or not frame.path.tolist() == historical.path.tolist():
        raise ValueError("Historical windows no longer align with waveform cache.")
    if not frame.split.tolist() == historical.split.tolist() or not np.allclose(frame.start_s, historical.start_s):
        raise ValueError("Historical partitions changed.")
    frame["wave_index"] = frame.index
    mapping_path = ROOT / "results/species_nested_2026_10_13/recording_predictions.csv"
    mapping = pd.read_csv(mapping_path)[["group", "collision_group"]].drop_duplicates()
    if mapping.groupby("group").collision_group.nunique().max() != 1:
        raise ValueError("Conflicting PCM collision identities.")
    frame = frame.merge(mapping, on="group", how="left", validate="many_to_one")
    if frame.collision_group.isna().any():
        raise ValueError("Every Dryad recording needs a PCM audit identity.")
    frame["pcm_block"] = frame.collision_group
    frame, removed = purge_partition_collisions(frame)
    frame = frame.loc[frame.split.ne("calibration")].copy().reset_index(drop=True)
    classes = sorted(frame.label.unique())
    frame["target"] = frame.label.map({name: i for i, name in enumerate(classes)})
    if frame.loc[frame.split.eq("train"), "target"].nunique() != len(classes):
        raise ValueError("Training does not contain all target classes.")
    cache = ROOT / "data/cache/torres_2025"
    cache.mkdir(exist_ok=True, parents=True)
    indices = frame.wave_index.to_numpy(dtype=int)
    key = hashlib.sha256(indices.tobytes() + b"mel32_fft1024_hop256_sr16000_v1").hexdigest()
    feature_path, cache_meta = cache / (key + ".npy"), cache / (key + ".json")
    frontend = MelSequence().to(device).eval()
    if not feature_path.exists():
        waves = np.load(ROOT / "data/cache/large_cnn/waves.npy", mmap_mode="r")
        with torch.inference_mode():
            first = frontend(torch.tensor(np.array(waves[indices[:1]]), device=device)).cpu().numpy()
            x = np.lib.format.open_memmap(feature_path, mode="w+", dtype="float32", shape=(len(frame), *first.shape[1:]))
            for start in range(0, len(frame), 256):
                value = torch.tensor(np.array(waves[indices[start:start + 256]]), device=device)
                x[start:start + len(value)] = frontend(value).cpu().numpy()
            x.flush()
        write_json(cache_meta, dict(rows=len(frame), key=key, frontend=frontend.config,
                                  waveform_samples=int(waves.shape[1]), feature_sha256=digest(feature_path)))
    meta = json.loads(cache_meta.read_text())
    if meta["key"] != key or digest(feature_path) != meta["feature_sha256"]:
        raise ValueError("Cached recurrent features failed verification.")
    x = np.load(feature_path, mmap_mode="r")
    metadata = dict(task="20-species classification", classes=classes, input_shape=list(x.shape),
        input_features=str(feature_path.relative_to(ROOT)), feature_sha256=meta["feature_sha256"],
        frontend=frontend.config, duration_s=15872 / 16000, source_partitions_sha256=digest(ROOT / "results/ifsc_melhorias_2026_10_11/partitions.csv"),
        pcm_mapping_sha256=digest(mapping_path), historical_test=True, unknown_individual_overlap=True,
        split_protocol="historical partitions with whole-PCM-component priority test > validation > calibration > train",
        removed_windows=len(removed), removed_groups=int(removed.group.nunique()),
        excluded_mosquitosong=True, infection_labels_available=False,
        comparison="fresh baseline fitted on identical retained training rows; older CNN/MLP scores not same-protocol comparators")
    return frame, x, metadata, removed


def weights_for(frame):
    groups = frame.groupby("target").group.nunique()
    windows = frame.groupby("group").size()
    w = 1 / (frame.target.map(groups).to_numpy() * frame.group.map(windows).to_numpy())
    return (w / w.mean()).astype("float32")


def validation_loss(frame, probabilities):
    _, y, p = aggregate_groups(frame, probabilities)
    counts = np.bincount(y, minlength=p.shape[1])
    weights = 1 / counts[y]
    return float(np.average(-np.log(p[np.arange(len(y)), y].clip(1e-12)), weights=weights))


def train_candidate(name, frame, x, classes, device, output, config):
    torch.manual_seed(SEED)
    model = TorresLSTM(classes=len(classes), bidirectional=name == "bilstm").to(device)
    train = np.flatnonzero(frame.split.eq("train"))
    validation = np.flatnonzero(frame.split.eq("validation"))
    values = torch.tensor(np.array(x[train]), device=device)
    labels = torch.tensor(frame.target.to_numpy()[train], dtype=torch.long, device=device)
    weights = torch.tensor(weights_for(frame.iloc[train]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=(0.9, 0.999))
    best_loss, stale, best_epoch = float("inf"), 0, 0
    history = []
    start = time.monotonic()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        order = torch.multinomial(weights, len(train), replacement=True)
        loss_sum, count = 0.0, 0
        for begin in range(0, len(order), config["batch_size"]):
            index = order[begin:begin + config["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(values[index]), labels[index])
            loss.backward()
            if config["gradient_clip"] is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            loss_sum += float(loss.detach()) * len(index)
            count += len(index)
        p = predict_probabilities(model, x[validation], device)
        vl = validation_loss(frame.iloc[validation], p)
        _, truth, grouped = aggregate_groups(frame.iloc[validation], p)
        row = dict(model=name, epoch=epoch, train_loss=loss_sum / count,
                   validation_recording_loss=vl,
                   validation_recording_accuracy=float((grouped.argmax(1) == truth).mean()))
        history.append(row)
        if vl < best_loss - 1e-4:
            best_loss, stale, best_epoch = vl, 0, epoch
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(dict(state_dict=state, config=model.config, classes=classes), output / f"{name}.pt")
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            log(task=output.name, **row, best_epoch=best_epoch, seconds=time.monotonic() - start)
            write_json(output / f"{name}_history.json", history)
        if stale >= config["patience"]:
            break
    checkpoint = torch.load(output / f"{name}.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    write_json(output / f"{name}_history.json", history)
    return model, dict(model=name, best_epoch=best_epoch, epochs_run=len(history),
        validation_recording_loss=best_loss, parameters=sum(p.numel() for p in model.parameters()),
        seconds=time.monotonic() - start, checkpoint_sha256=digest(output / f"{name}.pt"))


def evaluate(model, name, frame, x, classes, device, output, passes=100):
    deterministic = predict_probabilities(model, x, device)
    groups, y, deterministic = aggregate_groups(frame, deterministic)
    draws = monte_carlo_probabilities(model, x, device, passes=passes, seed=SEED)
    _, _, grouped = aggregate_groups(frame, draws)
    stats = vote_statistics(grouped)
    thresholds = [0, 0.01, 0.05, 0.10, 0.20, 0.50]
    report = dict(model=name, recordings=len(groups), windows=len(frame), mc_passes=passes,
        deterministic=classification_metrics(y, deterministic.argmax(1), classes),
        mc_mean=classification_metrics(y, stats["mean_prediction"], classes),
        mc_vote=classification_metrics(y, stats["prediction"], classes),
        selective=[selective_metrics(y, stats["prediction"], stats["uncertainty"], classes, threshold)
                   for threshold in thresholds],
        uncertainty_definition="1 - largest class vote count / MC passes; binary equivalent of paper UC/100",
        recording_aggregation="arithmetic mean of window probabilities within each draw",
        unanimous_does_not_mean_certainty=True)
    rows = pd.DataFrame(dict(group=groups, target=y, truth=np.asarray(classes)[y],
        deterministic_prediction=np.asarray(classes)[deterministic.argmax(1)],
        mc_prediction=np.asarray(classes)[stats["prediction"]],
        mc_mean_prediction=np.asarray(classes)[stats["mean_prediction"]],
        uncertainty=stats["uncertainty"], accepted_unanimous=stats["uncertainty"] <= 1e-12,
        confidence=stats["mean_probability"].max(1), predictive_entropy=stats["predictive_entropy"],
        mutual_information=stats["mutual_information"]))
    for i, label in enumerate(classes):
        rows[f"prob_{i}"] = stats["mean_probability"][:, i]
        rows[f"votes_{i}"] = stats["vote_counts"][:, i]
    rows.to_csv(output / f"{name}_predictions.csv", index=False)
    np.savez_compressed(output / f"{name}_mc_group_probabilities.npz", probabilities=grouped.astype("float32"), groups=groups, target=y)
    write_json(output / f"{name}_metrics.json", report)
    log(task=output.name, model=name, deterministic=report["deterministic"], unanimous=report["selective"][0])
    return report


def author_checkpoints(output, device, passes):
    output.mkdir()
    frame, x, metadata, _ = load_infection()
    _, test = train_test_split(np.arange(len(x)), test_size=40, random_state=42, stratify=frame.target)
    frame = frame.iloc[test].copy()
    frame["split"] = "author_checkpoint_selection_subset"
    frame.to_csv(output / "partitions.csv", index=False)
    metadata.update(scope="retrospective reexecution: public checkpoint selected on this same 40-example subset",
                    split_matches_public_code=True, differs_from_article_80_20=True,
                    article_mc_passes=100, author_testing_script_mc_passes=10,
                    implementation="equivalent stacked single-layer LSTMs with explicit inter-layer dropout; RNG is not author-identical")
    write_json(output / "protocol.json", metadata)
    reports = []
    for name, filename in [("lstm", "LSTM_Uni_2H_256.pth"), ("bilstm", "LSTM_Bid_2H_256.pth")]:
        model = TorresLSTM(bidirectional=name == "bilstm").to(device)
        path = AUTHOR / "Model" / filename
        model.load_author_state(torch.load(path, map_location="cpu", weights_only=True))
        row = evaluate(model, name, frame, x[test], metadata["classes"], device, output, passes)
        row["author_checkpoint_sha256"] = digest(path)
        reports.append(row)
    write_json(output / "metrics.json", reports)


def run_task(task, output, device, passes, epoch_override=None):
    output.mkdir()
    frame, x, metadata, removed = load_infection() if task == "infection" else load_species(device)
    classes = metadata["classes"]
    frame.to_csv(output / "partitions.csv", index=False)
    if len(removed):
        removed.to_csv(output / "purged_collisions.csv", index=False)
    tr, va, te = [np.flatnonzero(frame.split.eq(s)) for s in ["train", "validation", "test"]]
    config = dict(epochs=300 if task == "infection" else 80,
                  lr=0.000005 if task == "infection" else 0.0003,
                  batch_size=16 if task == "infection" else 128,
                  patience=50 if task == "infection" else 15,
                  gradient_clip=None if task == "infection" else 3)
    if epoch_override is not None:
        config["epochs"] = epoch_override
    # The fitted transformation is deliberately restricted to the training rows.
    if task == "species":
        mean = np.asarray(x[tr]).mean((0, 1), keepdims=True)
        std = np.asarray(x[tr]).std((0, 1), keepdims=True).clip(1)
        np.savez(output / "scaler.npz", mean=mean, std=std)
        x = ((x - mean) / std).astype("float32")
    metadata.update(training=config, seed=SEED, device=str(device), torch_version=str(torch.__version__),
        recurrent=dict(layers=2, hidden=256, dropout=0.2, readout="linear(last_output) as author code"),
        loss="CrossEntropyLoss as public author code; paper table says binary cross entropy",
        mc_passes=passes, uncertainty_thresholds_predeclared=[0, 0.01, 0.05, 0.1, 0.2, 0.5],
        selection="minimum class-balanced recording validation log loss, never test",
        split_counts={name: dict(windows=len(g), groups=int(g.group.nunique()), pcm_blocks=int(g.pcm_block.nunique()))
                      for name, g in frame.groupby("split")},
        author_repository=AUTHOR_REPOSITORY, author_commit=AUTHOR_COMMIT,
        implementation_sha256={name: digest(ROOT / name) for name in ["torres_lstm.py", "apply_torres_2025.py", "fetch_torres_2025.py"]})
    write_json(output / "protocol.json", metadata)
    # A fixed linear summary baseline is fitted afresh on identical training rows.
    summary = np.concatenate([np.asarray(x).mean(1), np.asarray(x).std(1)], axis=1)
    baseline = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=SEED))
    baseline.fit(summary[tr], frame.target.iloc[tr], logisticregression__sample_weight=weights_for(frame.iloc[tr]))
    baseline_validation_loss = validation_loss(frame.iloc[va], baseline.predict_proba(summary[va]))
    joblib.dump(baseline, output / "linear_baseline.joblib")
    models, candidates = {}, []
    for name in ["lstm", "bilstm"]:
        model, row = train_candidate(name, frame, x, classes, device, output, config)
        models[name] = model
        candidates.append(row)
        write_json(output / "training_summary.json", candidates)
    selected = min(candidates, key=lambda row: row["validation_recording_loss"])["model"]
    write_json(output / "freeze.json", dict(selected_neural_model=selected,
        criterion=metadata["selection"], candidates=candidates,
        linear_baseline_validation_loss=baseline_validation_loss,
        checkpoint_sha256={name: digest(output / f"{name}.pt") for name in models},
        baseline_sha256=digest(output / "linear_baseline.joblib"),
        scaler_sha256=digest(output / "scaler.npz") if task == "species" else None,
        frozen_before_test_inference=True))
    # First use of these test labels for model evaluation in this run starts here.
    _, y, bp = aggregate_groups(frame.iloc[te], baseline.predict_proba(summary[te]))
    baseline_metrics = classification_metrics(y, bp.argmax(1), classes)
    write_json(output / "linear_baseline_metrics.json", baseline_metrics)
    reports = [evaluate(models[name], name, frame.iloc[te], x[te], classes, device, output, passes)
               for name in models]
    write_json(output / "metrics.json", dict(selected_neural_model=selected,
        linear_baseline=baseline_metrics, neural_models=reports))


def main(args):
    if args.output.exists():
        raise FileExistsError("Existing results are preserved: " + str(args.output))
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    device = torch.device(args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
    tasks = ["author", "infection", "species"] if args.task == "all" else [args.task]
    source_dir = args.output / "source"
    source_dir.mkdir()
    for name in ["torres_lstm.py", "apply_torres_2025.py", "fetch_torres_2025.py"]:
        shutil.copy2(ROOT / name, source_dir / name)
    paper = ROOT / "literature/9221_torres_2025_dengue_infected_mosquito_detection.pdf"
    write_json(args.output / "run.json", dict(tasks=tasks, seed=SEED, device=str(device),
        article="Torres et al., IEEE Latin America Transactions 23(2), 135-143, 2025",
        author_repository=AUTHOR_REPOSITORY, author_commit=AUTHOR_COMMIT,
        article_pdf_sha256=digest(paper) if paper.exists() else None))
    for task in tasks:
        log(starting=task)
        if task == "author":
            author_checkpoints(args.output / "author_checkpoints", device, args.mc_passes)
        else:
            run_task(task, args.output / task, device, args.mc_passes, args.epochs)
    write_json(args.output / "complete.json", dict(complete=True, tasks=tasks))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["all", "author", "infection", "species"], default="all")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--mc-passes", type=int, default=100)
    parser.add_argument("--epochs", type=int, help="Explicit smoke/debug override; default uses the recorded task protocol")
    main(parser.parse_args())
