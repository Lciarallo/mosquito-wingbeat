"""Use a newly trained Torres-style model with 100 stochastic dropout passes.

Species models accept WAV/audio. Infection models accept the original author
mel tensors, whose raw-audio extraction recipe is not distributed in that repo.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torres_lstm import (MelSequence, TorresLSTM, aggregate_groups, digest,
                         monte_carlo_probabilities, vote_statistics, write_json)


def predict(model_directory, *, audio=None, features=None, author_row=None,
            model_name=None, passes=100, threshold=0, device="auto"):
    model_directory = Path(model_directory)
    protocol = json.loads((model_directory / "protocol.json").read_text())
    freeze = json.loads((model_directory / "freeze.json").read_text())
    name = model_name or freeze["selected_neural_model"]
    device = torch.device(device if device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
    if name == "linear":
        import joblib
        path = model_directory / "linear_baseline.joblib"
        if digest(path) != freeze["baseline_sha256"]:
            raise ValueError("Linear model differs from the frozen evaluation checkpoint.")
        model = joblib.load(path)
        classes = protocol["classes"]
        expected_features = protocol.get("frontend", {}).get("bands", 32)
    else:
        path = model_directory / f"{name}.pt"
        if digest(path) != freeze["checkpoint_sha256"][name]:
            raise ValueError("Model differs from the frozen evaluation checkpoint.")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = TorresLSTM(**checkpoint["config"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        classes = checkpoint["classes"]
        expected_features = model.config["features"]
    reference = None
    if audio is not None:
        if "frontend" not in protocol:
            raise ValueError("Infection model requires original-scale mel features; use --author-row or --features.")
        from ternary_protocol import read_audio
        wave = read_audio(Path(audio), sample_rate=16000)
        windows = len(wave) // 15872
        if not windows:
            raise ValueError("Audio must contain at least 0.992 seconds.")
        if "variant" in protocol["frontend"]:
            from species_preprocessing import SpeciesFrontend
            frontend = SpeciesFrontend(protocol["frontend"]["variant"]).to(device).eval()
        else:
            frontend = MelSequence().to(device).eval()
        with torch.inference_mode():
            x = np.concatenate([frontend(torch.tensor(wave[start * 15872:min(start + 256, windows) * 15872]
                        .reshape(-1, 15872).copy(), device=device)).cpu().numpy()
                                for start in range(0, windows, 256)])
    elif author_row is not None:
        if "frontend" in protocol:
            raise ValueError("Author infection features are not the species frontend.")
        from apply_torres_2025 import load_infection
        _, values, _, _ = load_infection()
        if not 0 <= author_row < len(values):
            raise ValueError("Author row is out of range.")
        x = np.array(values[author_row:author_row + 1])
        reference = dict(source="public author tensor", row=author_row)
    else:
        x = np.load(features, allow_pickle=False)
        if x.ndim == 2:
            x = x[None]
    if x.ndim != 3 or x.shape[-1] != expected_features or not len(x) or not np.isfinite(x).all():
        raise ValueError(f"Expected finite (windows, time, {expected_features}) features.")
    if (model_directory / "scaler.npz").exists():
        scaler_path = model_directory / "scaler.npz"
        if digest(scaler_path) != freeze["scaler_sha256"]:
            raise ValueError("Scaler differs from the frozen training transformation.")
        with np.load(scaler_path, allow_pickle=False) as scale:
            x = ((x - scale["mean"]) / scale["std"]).astype("float32")
    frame = pd.DataFrame({"group": ["input"] * len(x), "target": [0] * len(x)})
    if name == "linear":
        summary = np.concatenate([x.mean(1), x.std(1)], axis=1)
        _, _, probability = aggregate_groups(frame, model.predict_proba(summary))
        label = classes[int(probability[0].argmax())]
        return dict(model=name, task=protocol["task"], windows=len(x), monte_carlo_passes=0,
                    predicted_label=label, decision=label, accepted=True,
                    probabilities={c: float(p) for c, p in zip(classes, probability[0])}, reference=reference,
                    scope="forced class choice; linear model has no Monte Carlo dropout or presence rejection")
    draws = monte_carlo_probabilities(model, x, device, passes=passes)
    _, _, recording = aggregate_groups(frame, draws)
    stats = vote_statistics(recording)
    uncertainty = float(stats["uncertainty"][0])
    accepted = uncertainty <= threshold + 1e-12
    return dict(model=name, task=protocol["task"], windows=len(x), monte_carlo_passes=passes,
                predicted_label=classes[int(stats["prediction"][0])],
                decision=classes[int(stats["prediction"][0])] if accepted else "uncertain",
                accepted=accepted, disagreement_fraction=uncertainty,
                maximum_allowed_disagreement=threshold,
                probabilities={name: float(p) for name, p in zip(classes, stats["mean_probability"][0])},
                votes={name: int(count) for name, count in zip(classes, stats["vote_counts"][0])},
                reference=reference,
                scope="research model; MC agreement measures model disagreement, not confirmed biological status")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_directory", type=Path)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--audio", type=Path)
    inputs.add_argument("--features", type=Path, help="Unscaled mel sequences in the trained frontend's units")
    inputs.add_argument("--author-row", type=int)
    parser.add_argument("--model", choices=["lstm", "bilstm", "linear"])
    parser.add_argument("--passes", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 <= args.threshold < 1:
        parser.error("threshold must be between 0 and 1")
    result = predict(args.model_directory, audio=args.audio, features=args.features,
                     author_row=args.author_row, model_name=args.model, passes=args.passes,
                     threshold=args.threshold, device=args.device)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
