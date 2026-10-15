"""Audit original splitting by tracing waveform views to their source recordings."""
from bisect import bisect_right
import json
import os
from pathlib import Path
import hashlib
import numpy as np
import librosa

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "tmp/reproductions_2026_10_01/mosquitosongp/prediction.ipynb"
OUT = ROOT / "results/faithful_reproduction_2026_10_03/mosquitosong"
WAVS = ROOT / "data/external/additional/mosquitosong"


def main():
    nb = json.loads(SOURCE.read_text())
    ns = {"np": np, "os": os, "librosa": librosa}
    import math, re
    ns.update(math=math, re=re)
    for i in [2, 3, 15, 16, 17, 18, 19]:
        exec(compile("".join(nb["cells"][i]["source"]), str(SOURCE)+":cell"+str(i), "exec"), ns)
    names = ns["load_data"](str(WAVS))
    counts = ns["count_data"](names, str(WAVS)+"/")
    original_load = librosa.load
    reports = []
    for fold in range(10):
        loaded = []
        def trace_load(path, *a, **kw):
            x, sr = original_load(path, *a, **kw)
            loaded.append((Path(path).name, x))
            return x, sr
        librosa.load = trace_load
        try:
            parts = ns["split_data"](names, counts, fold, str(WAVS)+"/")[:3]
        finally:
            librosa.load = original_load
        addresses = sorted((x.__array_interface__["data"][0], x.nbytes, name, x.itemsize)
                           for name, x in loaded)
        starts = [a[0] for a in addresses]
        groups = {}
        for split, waves in zip(["train", "val", "test"], parts):
            files = {}
            for y in waves:
                ptr = y.__array_interface__["data"][0]
                idx = bisect_right(starts, ptr)-1
                base, nbytes, name, itemsize = addresses[idx]
                assert base <= ptr and ptr+y.nbytes <= base+nbytes
                start = (ptr-base)//itemsize
                files.setdefault(name, []).append((start, start+len(y)))
            groups[split] = files
        row = dict(fold=fold+1, windows={s: sum(map(len, files.values())) for s, files in groups.items()},
                   recordings={s: len(files) for s, files in groups.items()}, shared_recordings={},
                   cross_split_overlapping_window_pairs={})
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            shared = set(groups[a]) & set(groups[b])
            row["shared_recordings"][a+"_"+b] = len(shared)
            row["cross_split_overlapping_window_pairs"][a+"_"+b] = sum(
                max(x0,y0) < min(x1,y1)
                for name in shared for x0,x1 in groups[a][name] for y0,y1 in groups[b][name])
        reports.append(row)
        print("SPLIT AUDIT", row, flush=True)
    info = dict(source_notebook_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                source_functions_unchanged=True, method="Original split_data; librosa load traced without changing its return; numpy views mapped by memory address.",
                individual_mosquito_ids_available=False, folds=reports)
    (OUT / "split_audit.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
