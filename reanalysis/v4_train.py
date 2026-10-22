"""v4 development, selection and final freeze (reanalysis/protocol_v4.json, reports/plano_v4.md).

Commands: freeze-check | dev | select | final | diagnostics.
Only v3 development roles are loaded; the v3 reserved test is dropped before any computation.
Cluster split, calibration maps, threshold, few-shot cluster choice and metrics are imported unchanged
from v3 so that differences come from the pre-registered candidates, not from re-implementations.
"""
import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
from torch import nn

from reanalysis import v3_metrics as metrics
from reanalysis import v3_train as v3

ROOT = v3.ROOT
OUT = ROOT / "results/v4"
PUBLIC = ROOT / "reanalysis/outputs/v4"
AUG = ROOT / "data/cache/v4_augmented"
PROTOCOL = ROOT / "reanalysis/protocol_v4.json"
FREEZE = ROOT / "reanalysis/protocol_v4_freeze.json"
SEED = 20261101
DEV_ROLES = ["train", "noise_train"]
RESERVED = ["locked_test", "locked_noise", "quarantine_holdout_related"]
LODO_MOSQUITO = ["abuzz", "humbug", "mosquitosong"]
LODO_NOISE = ["esc50", "pete_noise"]
CANDIDATES = ["A0_perch_white", "A1_perch_coral", "A2_perch_dann", "B1_band_tabular", "B2_wingbeat",
              "C1_aug_band_tabular", "C2_aug_perch_coral", "D1_perch_ft"]  # D1: amendment A1
DIAGNOSTIC = ["R_tabular_full"]
SPEC = {"A0_perch_white": ("perch_white", "logistic", False, False),
        "A1_perch_coral": ("perch_white", "logistic", True, False),
        "A2_perch_dann": ("perch_white", "dann", False, False),
        "B1_band_tabular": ("band_tabular", "histgb", False, False),
        "B2_wingbeat": ("wingbeat", "logistic", False, False),
        "C1_aug_band_tabular": ("band_tabular", "histgb", False, True),
        "C2_aug_perch_coral": ("perch_white", "logistic", True, True),
        "D1_perch_ft": ("perch_ft", "logistic", False, False),
        "R_tabular_full": ("tabular_white", "histgb", False, False)}
FINETUNE = ROOT / "results/v4/finetune"
AMENDMENTS = [ROOT / "reanalysis/protocol_v4_amendment_A1_freeze.json",
              ROOT / "reanalysis/protocol_v4_amendment_A2_freeze.json"]
LOW_BANDS = [0, 1, 2, 3, 4]  # mel centres 133-276 Hz, below the Aedes wingbeat
BAND_COLUMNS = [b for b in range(40) if b not in LOW_BANDS] + [40 + b for b in range(40) if b not in LOW_BANDS] + list(range(80, 89))
FROZEN_FILES = ["reanalysis/protocol_v4.json", "reports/plano_v4.md", "reanalysis/v4_train.py",
                "reanalysis/v4_augment.py", "reanalysis/v4_diagnostics.py", "reanalysis/v4_finetune.py",
                "reanalysis/protocol_v4_amendment_A1.json", "reanalysis/protocol_v4_amendment_A2.json", "reanalysis/v3_train.py", "reanalysis/v3_metrics.py",
                "reanalysis/v3_features.py", "results/v3_continuation/manifest.csv",
                "data/cache/v3_continuation/meta.json"]


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=float) + "\n")


def check_freeze():
    """Refuse to run with code/protocol/data different from the pre-registered freeze.
    The original freeze is immutable; a later amendment freeze must list every file it changed and
    re-hash all frozen inputs. Files not declared as changed must still match the original freeze."""
    frozen = json.loads(FREEZE.read_text())
    declared = set()
    latest = frozen
    for path in AMENDMENTS:
        if path.exists():
            latest = json.loads(path.read_text())
            declared |= set(latest["changed_files"])
    stale = [p for p, sha in frozen["sha256"].items() if p not in declared and digest(ROOT / p) != sha]
    changed = [p for p, sha in latest["sha256"].items() if digest(ROOT / p) != sha]
    if stale or changed:
        raise ValueError(f"Frozen v4 inputs changed (record an amendment): {stale + changed}")
    return latest


# ----------------------------------------------------------------------------- data
def development_only(frame, arrays):
    """Drop every reserved row before anything else touches the arrays."""
    keep = frame.role.isin(DEV_ROLES).to_numpy()
    frame = frame.loc[keep].reset_index(drop=True)
    arrays = {k: v[keep] for k, v in arrays.items()}
    if frame.role.isin(RESERVED).any():
        raise AssertionError("reserved v3 rows in v4 development frame")
    arrays["band_tabular"] = arrays["tabular_white"][:, BAND_COLUMNS]
    return frame, arrays


def load_development():
    frame, arrays = v3.load()
    return development_only(frame, arrays)


def load_augmented(frame):
    meta = json.loads((AUG / "meta.json").read_text())
    if not meta.get("complete") or digest(AUG / "features.npz") != meta["features_sha256"]:
        raise ValueError("Incomplete or changed augmented cache")
    with np.load(AUG / "features.npz") as cache:
        position = pd.Series(np.arange(len(cache["file_ids"])), index=cache["file_ids"])
        order = position.loc[frame.file_id.to_numpy()].to_numpy()
        aug = {"perch_white": cache["perch_white"][order], "tabular_white": cache["tabular_white"][order]}
    aug["band_tabular"] = aug["tabular_white"][:, BAND_COLUMNS]
    return aug


def partitions(frame):
    """Outer LODO splits; training clusters that also occur in the held-out set are removed."""
    out = []
    for kind, held in [("mosquito", LODO_MOSQUITO), ("noise", LODO_NOISE)]:
        for corpus in held:
            test = np.flatnonzero(frame.origin.eq(corpus).to_numpy())
            touched = set(frame.cluster_merged.iloc[test])
            train = np.array([i for i in np.flatnonzero(~frame.origin.eq(corpus).to_numpy())
                              if frame.cluster_merged.iloc[i] not in touched], dtype=int)
            for key in ["path", "component", "cluster_merged"]:
                if key in frame and set(frame[key].iloc[train]) & set(frame[key].iloc[test]):
                    raise ValueError(f"Outer overlap in {corpus}: {key}")
            out.append(dict(tag=f"LODO_{corpus}", kind=kind, corpus=corpus, train=train, test=test))
    return out


# ----------------------------------------------------------------------------- representations
@dataclass
class Coral:
    """PCA-128 then per-origin second-order alignment to the fitting rows (labels never used)."""
    pca: object = None
    target_mean: np.ndarray = None
    target_root: np.ndarray = None
    origin_maps: dict = field(default_factory=dict)

    @staticmethod
    def _root(cov, inverse=False):
        w, v = np.linalg.eigh(cov)
        w = np.clip(w, 1e-6, None)
        return (v * (w ** (-.5 if inverse else .5))) @ v.T

    def fit(self, X_fit, seed):
        self.pca = PCA(min(128, X_fit.shape[0] - 1, X_fit.shape[1]), random_state=seed).fit(X_fit)
        Z = self.pca.transform(X_fit)
        self.target_mean = Z.mean(0)
        self.target_root = self._root(LedoitWolf().fit(Z).covariance_)
        return self

    def add_origin(self, name, X):
        Z = self.pca.transform(X)
        self.origin_maps[name] = (Z.mean(0), self._root(LedoitWolf().fit(Z).covariance_, inverse=True))

    def transform(self, X, origins):
        Z = self.pca.transform(X)
        out = np.empty_like(Z)
        for name in np.unique(origins):
            mean, inv_root = self.origin_maps[name]
            use = origins == name
            out[use] = (Z[use] - mean) @ inv_root @ self.target_root + self.target_mean
        return out.astype("float32")


def finetune_matrix(frame, tag, fit, cal, test):
    """Fold-specific fine-tuned embeddings (amendment A1). The backbone must have been trained on exactly
    the fitting rows of this fold: never on calibration or evaluation files."""
    meta = json.loads((FINETUNE / tag / "embeddings.json").read_text())
    if digest(FINETUNE / tag / "embeddings.npy") != meta["embeddings_sha256"]:
        raise ValueError(f"Changed fine-tuned embeddings: {tag}")
    trained = set(meta["training_file_ids"])
    if trained != set(frame.file_id.iloc[fit]):
        raise ValueError(f"Fine-tuning rows differ from the fold's fitting rows: {tag}")
    if trained & set(frame.file_id.iloc[np.r_[cal, test]]):
        raise ValueError(f"Fine-tuning touched calibration/evaluation files: {tag}")
    return np.load(FINETUNE / tag / "embeddings.npy")


def fold_matrix(name, frame, arrays, aug, fit, cal, test, seed, tag=None):
    """Feature matrix for every row of the fold, plus augmented fitting rows when the candidate uses them.
    CORAL statistics: training origins from their fit+cal rows, held-out origin from its unlabeled test rows."""
    feature, _, coral, augmented = SPEC[name]
    if feature == "perch_ft":
        return finetune_matrix(frame, tag, fit, cal, test), None, None
    X = arrays[feature]
    origins = frame.origin.to_numpy()
    extra = None
    if augmented:
        extra = aug[feature][fit]
    if not coral:
        return X, extra, None
    c = Coral().fit(X[fit], seed)
    train_side = np.r_[fit, cal]
    for o in np.unique(origins[train_side]):
        c.add_origin(o, X[train_side][origins[train_side] == o])
    for o in np.unique(origins[test]):
        if o not in c.origin_maps:
            c.add_origin(o, X[test][origins[test] == o])
    Z = np.zeros((len(X), c.pca.n_components_), dtype="float32")
    rows = np.r_[fit, cal, test]
    Z[rows] = c.transform(X[rows], origins[rows])
    if extra is not None:
        extra = c.transform(extra, origins[fit])
    return Z, extra, c


# ----------------------------------------------------------------------------- heads
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class DANNNet(nn.Module):
    def __init__(self, dim, n_domains, lam):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(dim, 256), nn.ReLU())
        self.dropout = nn.Dropout(.2)
        self.output = nn.Linear(256, 1)
        self.domain = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, max(n_domains, 2)))
        self.lam = lam

    def forward(self, x):
        h = self.hidden(x)
        return self.output(self.dropout(h)).squeeze(1), h


def fit_head(kind, param, X, y, origins, seed, multiplier=None):
    """v3.fit_head for logistic/histgb; DANN reuses v3.Head's 'mlp' inference path."""
    if kind != "dann":
        return v3.fit_head(kind, param, X, y, origins, seed, multiplier)
    head = v3.Head("mlp", float(param))
    if len(np.unique(y)) < 2:
        head.constant = float(logit(np.clip(np.mean(y), 1e-8, 1 - 1e-8)))
        return head
    weights = v3.balanced_weights(y, multiplier)
    head.scaler = StandardScaler().fit(X)
    Z = head.scaler.transform(X).astype("float32")
    torch.manual_seed(seed)
    domains = np.unique(origins, return_inverse=True)[1]
    head.model = DANNNet(X.shape[1], int(domains.max()) + 1, float(param))
    opt = torch.optim.AdamW(head.model.parameters(), lr=.001, weight_decay=.001)
    xt, yt = torch.from_numpy(Z), torch.from_numpy(y.astype("float32"))
    wt, dt = torch.from_numpy(weights.astype("float32")), torch.from_numpy(domains)
    rng = np.random.default_rng(seed)
    for _ in range(v3.EPOCHS):
        head.model.train()
        order = rng.permutation(len(X))
        for start in range(0, len(order), 256):
            ix = torch.from_numpy(order[start:start + 256])
            opt.zero_grad()
            raw, hidden = head.model(xt[ix])
            loss = (nn.functional.binary_cross_entropy_with_logits(raw, yt[ix], reduction="none") * wt[ix]).mean()
            if param:
                dom = head.model.domain(GradReverse.apply(hidden, float(param)))
                loss = loss + nn.functional.cross_entropy(dom, dt[ix])
            loss.backward()
            opt.step()
    head.model.eval()
    return head


GRIDS = {"logistic": [.01, .1, 1.], "histgb": [15, 31], "dann": [0., .3, 1.]}


def select_head(kind, X, y, clusters, origins, seed, multiplier=None, original=None):
    """3 inner cluster folds on fitting rows only; balanced accuracy at logit 0 (as v3).
    With augmentation, inner validation scores only original (non-augmented) rows."""
    original = np.ones(len(y), dtype=bool) if original is None else original
    splits = list(StratifiedGroupKFold(3, shuffle=True, random_state=v3.SEED + 1).split(X, y, clusters))
    scores = []
    for param in GRIDS[kind]:
        values = []
        for k, (a, b) in enumerate(splits):
            if set(clusters[a]) & set(clusters[b]):
                raise ValueError("Inner cluster overlap")
            m = None if multiplier is None else multiplier[a]
            head = fit_head(kind, param, X[a], y[a], origins[a], seed + k, m)
            b = b[original[b]]
            values.append(float(balanced_accuracy_score(y[b], head.logits(X[b]) >= 0)))
        scores.append(dict(param=float(param), score=float(np.mean(values))))
    best = max(scores, key=lambda r: r["score"])["param"]
    return fit_head(kind, best, X, y, origins, seed, multiplier), dict(chosen=best, inner_scores=scores)


@dataclass
class Bundle:
    candidate: str
    stage1: object
    stage2: object
    presence_map: object
    aedes_map: object
    tau: float
    info: dict

    def scores(self, X):
        presence = self.presence_map.apply(self.stage1.logits(X))
        return presence * self.aedes_map.apply(self.stage2.logits(X)), presence

    def predict(self, X):
        score, presence = self.scores(X)
        return np.where(presence < .5, 0, np.where(score >= self.tau, 2, 1)), score, presence

    def recalibrated(self, X, y):
        """Few-shot arm: refit only the probability maps and the threshold on local files."""
        import copy
        new = copy.deepcopy(self)
        new.presence_map = v3.calibrator(self.stage1.logits(X), (y > 0).astype(int))
        mosquito = y > 0
        new.aedes_map = v3.calibrator(self.stage2.logits(X[mosquito]), (y[mosquito] == 2).astype(int))
        score, presence = new.scores(X)
        new.tau = v3.threshold(y, score, presence)
        return new


def fit_bundle(name, frame, X, extra, fit, cal, seed):
    _, kind, _, _ = SPEC[name]
    y = frame.target.to_numpy(int)
    clusters, origins = frame.cluster_merged.to_numpy(), frame.origin.to_numpy()
    Xf, yf, cf, of, mult = X[fit], y[fit], clusters[fit], origins[fit], None
    orig = np.ones(len(fit), dtype=bool)
    if extra is not None:
        # Each original+augmented pair carries one unit of training mass.
        Xf, yf, cf, of = np.r_[Xf, extra], np.r_[yf, yf], np.r_[cf, cf], np.r_[of, of]
        mult, orig = np.full(len(yf), .5), np.r_[orig, np.zeros(len(extra), dtype=bool)]
    presence_kind = "histgb" if kind == "histgb" else "logistic"
    first, first_info = select_head(presence_kind, Xf, (yf > 0).astype(int), cf, of, seed, mult, orig)
    m = yf > 0
    second, second_info = select_head(kind, Xf[m], (yf[m] == 2).astype(int), cf[m], of[m], seed + 100,
                                      None if mult is None else mult[m], orig[m])
    presence_map = v3.calibrator(first.logits(X[cal]), (y[cal] > 0).astype(int))
    cal_m = cal[y[cal] > 0]
    aedes_map = v3.calibrator(second.logits(X[cal_m]), (y[cal_m] == 2).astype(int))
    bundle = Bundle(name, first, second, presence_map, aedes_map, 1.,
                    dict(stage1=first_info, stage2=second_info, seed=seed, augmented_rows=0 if extra is None else len(extra),
                         fit_file_ids=frame.file_id.iloc[fit].tolist(), calibration_file_ids=frame.file_id.iloc[cal].tolist()))
    score, presence = bundle.scores(X[cal])
    bundle.tau = v3.threshold(y[cal], score, presence)
    bundle.info.update(threshold=bundle.tau, calibration_counts={str(c): int((y[cal] == c).sum()) for c in [0, 1, 2]})
    return bundle


COLUMNS = ["file_id", "path", "target", "origin", "species", "cluster_merged", "role", "duration_s"]


def predictions(bundle, gate, frame, X, Xgate, rows, tag):
    pred, score, presence = bundle.predict(X[rows])
    gate_pred = gate.predict(Xgate[rows])[0]
    gated = pred.copy()
    if bundle.candidate != "B2_wingbeat":
        gated[(pred == 2) != (gate_pred == 2)] = 3
    out = frame.iloc[rows][[c for c in COLUMNS if c in frame]].copy()
    out["pred_ungated"], out["pred_gated"], out["score"], out["presence_score"] = pred, gated, score, presence
    out["threshold"], out["fold"], out["candidate"] = bundle.tau, tag, bundle.candidate
    return out


def as_metric_frame(pred, gated):
    frame = pred.copy()
    frame["pred"] = frame["pred_gated" if gated else "pred_ungated"]
    return frame


# ----------------------------------------------------------------------------- development
def run_fold(part, frame, arrays, aug, names, directory, seed):
    directory.mkdir(parents=True, exist_ok=True)
    fit, cal = v3.split_fit_calibration(frame, part["train"], seed)
    test = part["test"]
    write_json(directory / "partitions.json", dict(fit=frame.file_id.iloc[fit].tolist(),
               calibration=frame.file_id.iloc[cal].tolist(), test=frame.file_id.iloc[test].tolist()))
    Xgate, _, _ = fold_matrix("B2_wingbeat", frame, arrays, aug, fit, cal, test, seed)
    gate = None
    for name in ["B2_wingbeat"] + [n for n in names if n != "B2_wingbeat"]:
        path = directory / f"{name}.joblib"
        X, extra, coral = fold_matrix(name, frame, arrays, aug, fit, cal, test, seed, part["tag"])
        if path.exists():
            saved = json.loads((directory / f"{name}.info.json").read_text())
            if digest(path) != saved["model_sha256"]:
                raise ValueError(f"Changed cached model: {path}")
            bundle = joblib.load(path)
        else:
            started = time.monotonic()
            with threadpool_limits(8):
                bundle = fit_bundle(name, frame, X, extra, fit, cal, seed)
            joblib.dump(dict(bundle=bundle, coral=coral), path, compress=3)
            write_json(directory / f"{name}.info.json", dict(bundle.info, model_sha256=digest(path),
                       seconds=round(time.monotonic() - started, 1)))
            print("FIT", part["tag"], name, round(time.monotonic() - started, 1), "s", flush=True)
            bundle = joblib.load(path)
        bundle = bundle["bundle"]
        if name == "B2_wingbeat":
            gate = bundle
        destination = directory / f"{name}.csv"
        if not destination.exists():
            predictions(bundle, gate, frame, X, Xgate, test, part["tag"]).to_csv(destination, index=False)
    return fit, cal


def fewshot(part, frame, arrays, aug, names, directory, seed):
    y = frame.target.to_numpy(int)
    rows = []
    gate_bundle = joblib.load(directory / "B2_wingbeat.joblib")["bundle"]
    fit = frame.index[frame.file_id.isin(json.loads((directory / "partitions.json").read_text())["fit"])].to_numpy()
    cal = frame.index[frame.file_id.isin(json.loads((directory / "partitions.json").read_text())["calibration"])].to_numpy()
    Xgate, _, _ = fold_matrix("B2_wingbeat", frame, arrays, aug, fit, cal, part["test"], seed)
    for name in names:
        bundle = joblib.load(directory / f"{name}.joblib")["bundle"]
        X, _, _ = fold_matrix(name, frame, arrays, aug, fit, cal, part["test"], seed, part["tag"])
        for k in [5, 10, 20]:
            for repeat in range(5):
                chosen, evaluate, reason = v3.choose_fewshot(frame, part["test"], k, v3.SEED + repeat)
                record = dict(corpus=part["corpus"], candidate=name, k=k, repeat=repeat)
                if reason:
                    rows.append(dict(record, available=False, reason=reason))
                    continue
                local = bundle.recalibrated(X[chosen], y[chosen])
                local_gate = gate_bundle.recalibrated(Xgate[chosen], y[chosen])
                p = predictions(local, local_gate, frame, X, Xgate, evaluate, f"{part['tag']}_k{k}_r{repeat}")
                for gated in [False, True]:
                    rows.append(dict(record, gated=gated, available=True, n_calibration=len(chosen),
                                     **{key: value for key, value in metrics.point(as_metric_frame(p, gated)).items()
                                        if key not in ["confusion", "class_clusters"]}))
    return rows


def development():
    check_freeze()
    frame, arrays = load_development()
    aug = load_augmented(frame)
    names = CANDIDATES + DIAGNOSTIC
    OUT.mkdir(parents=True, exist_ok=True)
    all_metrics, few = {}, []
    for k, part in enumerate(partitions(frame)):
        directory = OUT / "development" / part["tag"]
        run_fold(part, frame, arrays, aug, names, directory, SEED + k)
        for name in names:
            p = pd.read_csv(directory / f"{name}.csv")
            for gated in [False, True]:
                key = f"{name}|{'gated' if gated else 'ungated'}"
                all_metrics.setdefault(part["tag"], {})[key] = metrics.with_uncertainty(as_metric_frame(p, gated))
        if part["kind"] == "mosquito":
            few += fewshot(part, frame, arrays, aug, CANDIDATES, directory, SEED + k)
        print("DONE", part["tag"], flush=True)
    write_json(OUT / "development_metrics.json", all_metrics)
    pd.DataFrame(few).to_csv(OUT / "fewshot_calibration.csv", index=False)


# ----------------------------------------------------------------------------- selection
def select(all_metrics, few):
    rows = []
    for name in CANDIDATES:
        for gated in [False, True]:
            key = f"{name}|{'gated' if gated else 'ungated'}"
            mos = [all_metrics[f"LODO_{c}"][key]["point"] for c in LODO_MOSQUITO]
            noise = [all_metrics["LODO_humbug"][key]["point"]] + [all_metrics[f"LODO_{c}"][key]["point"] for c in LODO_NOISE]
            recall = [p["aedes_recall"] for p in mos]
            other = [p["fpr_other"] for p in mos if p["fpr_other"] is not None]
            noise_fpr = [p["fpr_noise"] for p in noise if p["fpr_noise"] is not None]
            noise_ok = all(v <= .01 for v in noise_fpr)
            eligible0 = all(r is not None and r >= .2 for r in recall) and all(v <= .02 for v in other) and noise_ok
            eligible_k = None
            for k in [5, 10, 20]:
                ok = noise_ok
                for c in LODO_MOSQUITO:
                    sub = few[(few.candidate == name) & (few.corpus == c) & (few.k == k) & few.available.astype(bool)]
                    sub = sub[sub.gated.astype(bool) == gated]
                    if sub.empty:
                        ok = False
                        break
                    ok &= bool(sub.aedes_recall.median() >= .2 and sub.fpr_other.median() <= .02)
                if ok:
                    eligible_k = k
                    break
            mean = lambda key_, default: float(np.mean([p[key_] if p[key_] is not None else default for p in mos]))
            rows.append(dict(candidate=name, gated=gated, eligible_k0=eligible0, eligible_k=eligible_k,
                             recall_per_corpus=dict(zip(LODO_MOSQUITO, recall)),
                             fpr_other_per_corpus=dict(zip(LODO_MOSQUITO, [p["fpr_other"] for p in mos])),
                             fpr_noise=dict(zip(["LODO_humbug"] + [f"LODO_{c}" for c in LODO_NOISE], [p["fpr_noise"] for p in noise])),
                             precision=mean("aedes_precision", 0.), auroc=mean("auroc", .5),
                             auroc_per_corpus=dict(zip(LODO_MOSQUITO, [p["auroc"] for p in mos]))))
    zero = [r for r in rows if r["eligible_k0"]]
    shot = sorted({r["eligible_k"] for r in rows if r["eligible_k"]})
    if zero:
        pool, level = zero, "k=0"
    elif shot:
        pool, level = [r for r in rows if r["eligible_k"] == shot[0]], f"k={shot[0]}"
    else:
        pool, level = rows, None
    key = (lambda r: (r["precision"], r["auroc"])) if level else (lambda r: r["auroc"])
    chosen = max(pool, key=key)
    return dict(candidate=chosen["candidate"], gated=chosen["gated"], eligible=level is not None, eligibility_level=level,
                reason="eligible under the frozen per-corpus targets" if level else
                       "No candidate met the per-corpus development targets; highest mean LODO AUROC frozen as descriptive fallback",
                candidates=rows)


def selection_stage():
    check_freeze()
    all_metrics = json.loads((OUT / "development_metrics.json").read_text())
    few = pd.read_csv(OUT / "fewshot_calibration.csv")
    result = select(all_metrics, few)
    write_json(OUT / "selection.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "candidates"}, indent=2), flush=True)


def final_stage():
    """Refit the selected candidate on all development files and freeze it before any new-test data exists."""
    check_freeze()
    selection = json.loads((OUT / "selection.json").read_text())
    frame, arrays = load_development()
    aug = load_augmented(frame)
    every = np.arange(len(frame))
    fit, cal = v3.split_fit_calibration(frame, every, SEED + 99)
    directory = OUT / "final"
    directory.mkdir(parents=True, exist_ok=True)
    empty = np.array([], dtype=int)
    stored = {}
    for name in sorted({selection["candidate"], "B2_wingbeat"}):
        X, extra, coral = fold_matrix(name, frame, arrays, aug, fit, cal, empty, SEED + 99, "final")
        with threadpool_limits(8):
            bundle = fit_bundle(name, frame, X, extra, fit, cal, SEED + 99)
        path = directory / f"{name}.joblib"
        joblib.dump(dict(bundle=bundle, coral=coral), path, compress=3)
        stored[name] = digest(path)
    freeze = dict(frozen_utc=pd.Timestamp.now("UTC").isoformat(), selection=selection["candidate"],
                  gated=selection["gated"], eligible=selection["eligible"], model_sha256=stored,
                  protocol_freeze_sha256=digest(FREEZE), new_test_data_read=False,
                  note="Scored once on a new test that meets protocol_v4.json new_test.eligibility; until then the claim is inconclusive.")
    write_json(directory / "final_freeze.json", freeze)
    print(json.dumps(freeze, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze-check", "dev", "select", "final", "diagnostics"])
    stage = parser.parse_args().stage
    if stage == "freeze-check":
        print(json.dumps(check_freeze(), indent=2))
    elif stage == "dev":
        development()
    elif stage == "select":
        selection_stage()
    elif stage == "final":
        final_stage()
    else:
        from reanalysis import v4_diagnostics
        v4_diagnostics.main()
