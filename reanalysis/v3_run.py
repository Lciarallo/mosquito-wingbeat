"""Resume the v3 continuation without selecting or retuning on the final test."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from reanalysis.v3_train import CACHE, OUT, ROOT, digest, write_json


def main():
    logs = ROOT / "output/v3_continuation"
    logs.mkdir(parents=True, exist_ok=True)
    code_paths = [f"reanalysis/{name}.py" for name in
                  ["v3_train", "v3_metrics", "v3_report", "v3_predict", "v3_run"]]
    protected = [ROOT / p for p in ["notebooks/detector_ternario_treinamento_teste_aplicacao.ipynb",
                                   "data/metadata/pattern_classifier_reference.json",
                                   "data/metadata/ternary_neural_reference.json",
                                   "reanalysis/protocol_v3.json", "reanalysis/protocol_v3_freeze.json"]]
    protected += list((ROOT / "results/pattern_classifiers_2026_10_15/final").glob("*"))
    frozen = dict(code_sha256={p: digest(ROOT / p) for p in code_paths},
                  protected_sha256={str(p.relative_to(ROOT)): digest(p) for p in protected if p.is_file()},
                  protocol_sha256=digest(ROOT / "reanalysis/protocol_v3_continuation.json"))
    freeze_path = OUT / "execution_freeze.json"
    if freeze_path.exists() and json.loads(freeze_path.read_text()) != frozen:
        raise ValueError("Code or protected artifacts changed from the frozen execution")
    write_json(freeze_path, frozen)
    def status(phase, **extra):
        record = dict(phase=phase, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
        write_json(OUT / "status.json", record)
        print(json.dumps(record), flush=True)
    while not (CACHE / "meta.json").exists():
        progress_path = CACHE / "progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        status("waiting_features", **progress)
        if progress_path.exists() and time.time() - progress_path.stat().st_mtime > 600:
            raise RuntimeError("Feature extraction has not progressed for ten minutes; inspect its log before resuming")
        time.sleep(15)
    environment = os.environ.copy()
    environment.update(KMP_AFFINITY="disabled", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for phase in ["dev", "final", "fewshot", "report"]:
        if any(digest(ROOT / p) != sha for p, sha in frozen["code_sha256"].items()):
            raise ValueError("Frozen execution code changed")
        status(phase)
        module = "reanalysis.v3_report" if phase == "report" else "reanalysis.v3_train"
        command = [sys.executable, "-m", module] + ([] if phase == "report" else [phase])
        with (logs / f"{phase}.log").open("a") as log:
            completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            status("failed", failed_phase=phase, log=str(logs / f"{phase}.log"), returncode=completed.returncode)
            raise SystemExit(completed.returncode)
    for path, sha in frozen["protected_sha256"].items():
        if digest(ROOT / path) != sha:
            raise ValueError(f"Protected historical artifact changed: {path}")
    result = json.loads((OUT / "final_metrics.json").read_text())
    status("complete", selected=result["selection"]["candidate"], success=result["success"],
           protected_artifacts_unchanged=True)


if __name__ == "__main__":
    main()
