"""Run the original DR-BioL Evaluation.py and author weights on its exact data.

Only DAT model selection and a sklearn positional-argument compatibility adapter
are needed. Dropout, inputs, batch size, argmax and author plots remain original.
Five predeclared seeds quantify evaluation dropout, not five training repetitions.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch
from sklearn import metrics
from sklearn.utils import class_weight

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "tmp/reproductions_2026_10_01/dr_biol"
DATA = ROOT / "data/external/faithful_2026_10_03/dr_biol"
OUT = ROOT / "results/faithful_reproduction_2026_10_03/dr_biol"
WORK = ROOT / "tmp/faithful_reproduction_2026_10_03/dr_author_work"
SEEDS = [0, 1, 2, 3, 4]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def binary_stats(y, p):
    tn, fp, fn, tp = metrics.confusion_matrix(y, p, labels=[False, True]).ravel()
    return dict(tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
                recall=float(tp/(tp+fn)) if tp+fn else None,
                specificity=float(tn/(tn+fp)) if tn+fp else None,
                precision=float(tp/(tp+fp)) if tp+fp else None)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    original = SOURCE / "application/Evaluation.py"
    source_text = original.read_text()
    # sklearn >=1.0 made this existing keyword-only; same weight formula.
    original_weights_function = class_weight.compute_class_weight
    def compatible_class_weights(weight, **kw):
        return original_weights_function(class_weight=weight, **kw)
    class_weight.compute_class_weight = compatible_class_weights
    torch.set_num_threads(4)
    all_rows = []
    started = time.time()
    for model_name in ["DR_BioL_CNN", "DAT_CNN"]:
        for seed in SEEDS:
            work = WORK / model_name / ("seed%d" % seed)
            work.mkdir(parents=True, exist_ok=True)
            for name, source in [("Dataset", DATA),
                                 ("Pretrained_models", SOURCE / "application/Pretrained_models")]:
                dest = work / name
                if not dest.exists():
                    dest.symlink_to(source, target_is_directory=True)
            os.chdir(work)
            namespace = {"__name__": "author_evaluation", "__file__": str(original)}
            code = source_text
            if model_name == "DAT_CNN":
                assert code.count("model_name = 'DR_BioL_CNN'") == 1
                code = code.replace("model_name = 'DR_BioL_CNN'", "model_name = 'DAT_CNN'")
            exec(compile(code, str(original), "exec"), namespace)
            saved = {}
            original_generator = namespace["DataGenerator_4domains_8species"]
            class RecordedGenerator(original_generator):
                def __init__(self, *a, **kw):
                    super().__init__(*a, **kw)
                    saved["domains"] = self.y_test_domain.copy()
                    if not (OUT / "data_audit.json").exists():
                        info = {}
                        hashes = {}
                        for split in ["train", "val", "test"]:
                            x = getattr(self, "X_" + split)
                            y = getattr(self, "y_" + split)
                            d = getattr(self, "y_" + split + "_domain")
                            hs = Counter(hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in x)
                            hashes[split] = set(hs)
                            info[split] = dict(shape=list(x.shape), dtype=str(x.dtype),
                                               species_counts=dict(Counter(map(int, y))),
                                               domain_counts=dict(Counter(map(int, d))),
                                               duplicate_feature_rows=len(x)-len(hs))
                        info["identical_feature_overlap"] = {"train_val": len(hashes["train"] & hashes["val"]),
                            "train_test": len(hashes["train"] & hashes["test"]),
                            "val_test": len(hashes["val"] & hashes["test"])}
                        info["recording_ids_available"] = False
                        info["limitation"] = "Identical feature hashes do not establish recording-level independence."
                        (OUT / "data_audit.json").write_text(json.dumps(info, indent=2))
            namespace["DataGenerator_4domains_8species"] = RecordedGenerator
            original_accuracy = namespace["accuracy_score"]
            def capture_accuracy(y, p):
                saved["y"] = np.array(y, copy=True)
                saved["p"] = np.array(p, copy=True)
                return original_accuracy(y, p)
            namespace["accuracy_score"] = capture_accuracy
            original_softmax = namespace["softmax"]
            def capture_softmax(x):
                saved["logits"] = x.copy()
                output = original_softmax(x)
                saved["probabilities"] = output.copy()
                return output
            namespace["softmax"] = capture_softmax
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            tick = time.time()
            print("AUTHOR", model_name, "seed", seed, "START", flush=True)
            namespace["main"]([])
            y, p, prob, domains = [saved[k] for k in ["y", "p", "probabilities", "domains"]]
            assert len(y) == 10256 and prob.shape == (10256, 8)
            onehot = np.eye(8)[y.astype(int)]
            row = dict(model=model_name, seed=seed, seconds=time.time()-tick,
                       n=len(y), accuracy=float(metrics.accuracy_score(y, p)),
                       macro_f1=float(metrics.f1_score(y, p, average="macro")),
                       roc_auc_micro=float(metrics.roc_auc_score(onehot.ravel(), prob.ravel())),
                       average_precision_micro=float(metrics.average_precision_score(onehot, prob, average="micro")),
                       binary=binary_stats(y==2, p==2),
                       domains={str(d): dict(n=int((domains==d).sum()),
                           accuracy=float(metrics.accuracy_score(y[domains==d], p[domains==d])),
                           binary=binary_stats(y[domains==d]==2, p[domains==d]==2)) for d in np.unique(domains)},
                       classification_report=metrics.classification_report(y, p,
                           target_names=namespace["config"].all_species_labels, output_dict=True, zero_division=0))
            all_rows.append(row)
            np.savez_compressed(OUT / ("%s_seed%d.npz" % (model_name, seed)),
                                y=y, p=p, probabilities=prob, domains=domains)
            (OUT / "metrics.json").write_text(json.dumps(all_rows, indent=2))
            print("AUTHOR", model_name, "seed", seed, "accuracy", row["accuracy"], flush=True)
            del namespace, saved, y, p, prob, domains
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    execution = dict(status="complete", training_rerun=False, seconds=time.time()-started,
                     seeds=SEEDS, seed_scope="five MC dropout evaluation draws of one checkpoint per model",
                     python=platform.python_version(), torch=torch.__version__,
                     device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                     source_commit=subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip(),
                     original_evaluation_sha256=sha(original),
                     original_framework_sha256={str(p.relative_to(SOURCE)): sha(p) for p in (SOURCE/"framework").glob("*.py")},
                     runtime_compatibility=["compute_class_weight positional-to-keyword adapter"],
                     source_edits=["DAT run only: model_name literal changed to authors' alternative DAT_CNN"],
                     dataset_record="10.5281/zenodo.17287037",
                     archive_checksum="md5:4d3fe3742d104a022bbea013b4bd290e",
                     dataset_pickle_sha256=sha(DATA / "Mosquito_sound_8species_4domains.pickle"),
                     models={p.name: sha(p) for p in (SOURCE / "application/Pretrained_models").glob("*.pth")})
    (OUT / "execution.json").write_text(json.dumps(execution, indent=2))
    print("AUTHOR EVALUATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
