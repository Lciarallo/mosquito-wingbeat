"""Execute every original notebook cell, unchanged, with original H5 weights.

Run with the isolated TensorFlow 2.4 interpreter. Instrumentation records the
arguments to classification_report; it does not change predictions or metrics.
"""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "tmp/reproductions_2026_10_01/mosquitosongp"
OUT = ROOT / "results/faithful_reproduction_2026_10_03/mosquitosong"
WORK = ROOT / "tmp/faithful_reproduction_2026_10_03/ms_author_work"
MODELS = ROOT / "data/external/faithful_2026_10_03/mosquitosong_models"
NOISE = ROOT / "data/external/faithful_2026_10_03/mosquitosong_data/environmental_noise_recordings"
WAVS = ROOT / "data/external/additional/mosquitosong"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Tee(io.TextIOBase):
    def __init__(self, buffer, target):
        self.buffer = buffer
        self.target = target

    def write(self, text):
        self.buffer.write(text)
        self.target.write(text)
        self.target.flush()
        return len(text)

    def flush(self):
        self.target.flush()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "data").mkdir(exist_ok=True)
    for dest, src in [(WORK / "models", MODELS),
                      (WORK / "data/mosquito_recordings", WAVS),
                      (WORK / "data/environmental_noise_recordings", NOISE),
                      (WORK / "classification_results", OUT / "author_csv")]:
        if src == OUT / "author_csv":
            src.mkdir(exist_ok=True)
        if not dest.exists():
            dest.symlink_to(src, target_is_directory=True)
    assert len(list(WAVS.glob("*.wav"))) == 1324
    assert len(list(NOISE.glob("*.wav"))) == 12
    assert len(list(MODELS.glob("*.h5"))) == 10
    original = SOURCE / "prediction.ipynb"
    nb = json.loads(original.read_text())
    original_sources = ["".join(c["source"]) for c in nb["cells"]]
    namespace = {"__name__": "__main__"}
    captures = []
    records = []

    def instrument():
        import numpy as np
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
        author_report = namespace["classification_report"]

        def record_report(y_true, y_pred, **kwargs):
            report = author_report(y_true, y_pred, **kwargs)
            fold = int(namespace["fold_round"])
            gain_index = int(namespace["idx"])
            metric_index = int(namespace["metric_index"])
            mapping = {str(k): int(v) for k, v in namespace["label_dict"].items()}
            if metric_index == 0:
                pos = [v for k, v in mapping.items() if k.startswith("Ae.Aegypti_")]
                y_binary = np.isin(y_true, pos)
                p_binary = np.isin(y_pred, pos)
                cm = confusion_matrix(y_binary, p_binary, labels=[False, True])
                tn, fp, fn, tp = [int(v) for v in cm.ravel()]
                row = dict(fold=fold, gain_index=gain_index,
                           model_filename=namespace["model_filename"],
                           n=len(y_true), labels=mapping,
                           accuracy=float(accuracy_score(y_true, y_pred)),
                           macro_f1=float(f1_score(y_true, y_pred, average="macro")),
                           species_accuracy=float(accuracy_score(np.asarray(y_true)//2, np.asarray(y_pred)//2)),
                           binary_confusion=cm.tolist(),
                           binary_recall=tp/(tp+fn), binary_specificity=tn/(tn+fp),
                           binary_precision=tp/(tp+fp) if tp+fp else 0,
                           report=report)
                captures.append(row)
                np.savez_compressed(str(OUT / ("fold%d_gain%d.npz" % (fold+1, gain_index))),
                                    y_true=np.asarray(y_true), y_pred=np.asarray(y_pred))
                (OUT / "metrics.json").write_text(json.dumps(captures, indent=2))
            return report

        namespace["classification_report"] = record_report

    os.chdir(WORK)
    started = time.time()
    count = 0
    try:
        for index, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            count += 1
            code = "".join(cell["source"])
            cell["outputs"] = []
            cell["execution_count"] = count
            buffer = io.StringIO()
            print("ORIGINAL CELL", index, "START", flush=True)
            tick = time.time()
            with contextlib.redirect_stdout(Tee(buffer, sys.stdout)):
                exec(compile(code, str(original) + ":cell" + str(index), "exec"), namespace)
            cell["outputs"] = [{"name": "stdout", "output_type": "stream", "text": buffer.getvalue()}]
            records.append(dict(cell=index, sha256=hashlib.sha256(code.encode()).hexdigest(),
                                seconds=time.time()-tick))
            if index == 29:
                instrument()
            print("ORIGINAL CELL", index, "END", flush=True)
            (OUT / "author_executed.ipynb").write_text(json.dumps(nb))
        assert ["".join(c["source"]) for c in nb["cells"]] == original_sources
    except BaseException as e:
        cell["outputs"] = [{"output_type": "error", "ename": type(e).__name__,
                            "evalue": str(e), "traceback": traceback.format_exc().splitlines()}]
        (OUT / "author_failed.ipynb").write_text(json.dumps(nb))
        raise
    import tensorflow as tf
    import numpy as np
    import librosa
    execution = dict(status="complete", training_rerun=False,
                     seconds=time.time()-started, python=platform.python_version(),
                     tensorflow=tf.__version__, numpy=np.__version__, librosa=librosa.__version__,
                     source_commit=subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip(),
                     source_notebook_sha256=sha(original), code_cells=records,
                     source_code_cells_unchanged=True,
                     instrumentation="classification_report arguments recorded; original return unchanged",
                     input_wav_count=1324, noise_wav_count=12,
                     models={p.name: sha(p) for p in MODELS.glob("*.h5")},
                     noise={p.name: sha(p) for p in NOISE.glob("*.wav")},
                     recordings={p.name: sha(p) for p in WAVS.glob("*.wav")})
    (OUT / "execution.json").write_text(json.dumps(execution, indent=2))
    print("AUTHOR NOTEBOOK COMPLETE", len(captures), "evaluations", flush=True)


if __name__ == "__main__":
    main()
