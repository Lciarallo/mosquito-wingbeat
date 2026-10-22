"""Research inference for the frozen selected v3 continuation model."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort

from reanalysis import v3_features as features
from reanalysis.v3_train import OUT, digest


def main(audio, threads=4):
    freeze = json.loads((OUT / "final_freeze.json").read_text())
    name = freeze["selection"]["candidate"]
    for model, sha in freeze["model_sha256"].items():
        if digest(OUT / "final" / f"{model}.joblib") != sha:
            raise ValueError("Model differs from the frozen final configuration")
    if digest(features.MODEL) != json.loads((features.OUT / "meta.json").read_text())["model_sha256"]:
        raise ValueError("Perch weights differ from the trained feature extractor")
    bundle = joblib.load(OUT / "final" / f"{name}.joblib")
    gate = joblib.load(OUT / "final/C_wingbeat.joblib")
    _, plain, white, tab_plain, tab_white = features.prepare_audio((0, str(Path(audio).resolve())))
    arrays = dict(wingbeat=tab_white[None, 80:89], tabular_white=tab_white[None])
    if bundle.feature.startswith("perch"):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(features.MODEL), options, providers=["CPUExecutionProvider"])
        chunks = white if bundle.feature.endswith("white") else plain
        vectors = session.run(["embedding"], {"inputs": np.stack(chunks)})[0]
        arrays[bundle.feature] = vectors.mean(0, keepdims=True)
    idx = np.array([0])
    pred, score, presence = bundle.predict(arrays, idx)
    gate_pred, _, _ = gate.predict(arrays, idx)
    if name != "C_wingbeat" and (pred[0] == 2) != (gate_pred[0] == 2):
        pred[0] = 3
    result = dict(label=["no_mosquito", "other_mosquito", "aedes_aegypti", "uncertain"][pred[0]],
                  aedes_score=float(score[0]), presence_score=float(presence[0]), threshold=bundle.tau,
                  selected_model=name, development_eligible=freeze["selection"]["eligible"],
                  field_validated=False, scores_are_not_field_probabilities=True)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    main(args.audio, args.threads)
