"""Read-only verification of dataset isolation, frozen weights and MC outcomes."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from torres_lstm import classification_metrics, digest, selective_metrics, vote_statistics, write_json


def verify(root):
    root = Path(root)
    complete = json.loads((root / "complete.json").read_text())
    assert complete["complete"]
    result = dict(complete=True, tasks={}, checks=[])
    for task in ["author_checkpoints", "infection", "species"]:
        directory = root / task
        if not directory.exists():
            continue
        protocol = json.loads((directory / "protocol.json").read_text())
        frame = pd.read_csv(directory / "partitions.csv")
        classes = protocol["classes"]
        if task != "author_checkpoints":
            for column in ["group", "pcm_block", "content_sha256"]:
                assert frame.groupby(column).split.nunique().max() == 1, (task, column)
            freeze = json.loads((directory / "freeze.json").read_text())
            assert freeze["frozen_before_test_inference"]
            candidates = freeze["candidates"]
            assert min(candidates, key=lambda c: c["validation_recording_loss"])["model"] == freeze["selected_neural_model"]
            for name, sha in freeze["checkpoint_sha256"].items():
                assert digest(directory / f"{name}.pt") == sha
                history = json.loads((directory / f"{name}_history.json").read_text())
                row = next(c for c in candidates if c["model"] == name)
                chosen = next(h for h in history if h["epoch"] == row["best_epoch"])
                assert abs(chosen["validation_recording_loss"] - row["validation_recording_loss"]) < 1e-12
            if freeze["scaler_sha256"]:
                assert digest(directory / "scaler.npz") == freeze["scaler_sha256"]
            assert digest(directory / "linear_baseline.joblib") == freeze["baseline_sha256"]
            for name, sha in protocol["implementation_sha256"].items():
                assert digest(root / "source" / name) == sha
        models = {}
        for name in ["lstm", "bilstm"]:
            metric = json.loads((directory / f"{name}_metrics.json").read_text())
            pred = pd.read_csv(directory / f"{name}_predictions.csv")
            with np.load(directory / f"{name}_mc_group_probabilities.npz", allow_pickle=False) as saved:
                probabilities, target, groups = saved["probabilities"], saved["target"], saved["groups"]
            assert len(probabilities) == 100
            np.testing.assert_array_equal(groups, pred.group.astype(str))
            np.testing.assert_array_equal(target, pred.target)
            np.testing.assert_allclose(probabilities.sum(-1), 1, atol=1e-5)
            stats = vote_statistics(probabilities)
            np.testing.assert_array_equal(stats["vote_counts"], pred[[f"votes_{i}" for i in range(len(classes))]])
            np.testing.assert_allclose(stats["mean_probability"], pred[[f"prob_{i}" for i in range(len(classes))]], atol=1e-6)
            np.testing.assert_allclose(stats["uncertainty"], pred.uncertainty, atol=1e-12)
            np.testing.assert_array_equal(np.asarray(classes)[stats["prediction"]], pred.mc_prediction)
            recalc = classification_metrics(target, stats["prediction"], classes)
            assert recalc["confusion_matrix"] == metric["mc_vote"]["confusion_matrix"]
            assert abs(recalc["accuracy"] - metric["mc_vote"]["accuracy"]) < 1e-12
            deterministic = classification_metrics(target, [classes.index(label) for label in pred.deterministic_prediction], classes)
            assert deterministic["confusion_matrix"] == metric["deterministic"]["confusion_matrix"]
            for entry in metric["selective"]:
                actual = selective_metrics(target, stats["prediction"], stats["uncertainty"], classes, entry["threshold"])
                assert actual["confusion_with_abstention"] == entry["confusion_with_abstention"]
                assert actual["accepted"] + actual["abstained"] == len(target)
                assert actual["correct"] <= actual["accepted"]
                assert actual["accuracy"] == entry["accuracy"]
            if task != "author_checkpoints":
                assert set(groups) == set(frame.loc[frame.split.eq("test"), "group"].astype(str))
            models[name] = dict(total=len(target), passes=len(probabilities), valid=True)
        result["tasks"][task] = models
    if "infection" in result["tasks"] and "species" in result["tasks"]:
        assert result["tasks"]["infection"]["lstm"]["total"] == 60
        assert result["tasks"]["species"]["lstm"]["total"] == 190
    result["checks"] = ["group/PCM/hash isolation", "frozen checkpoint digests", "validation-only model selection",
                        "100-draw probabilities", "saved prediction agreement", "coverage/abstention denominators"]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = verify(args.root)
    write_json(args.root / "verification.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
