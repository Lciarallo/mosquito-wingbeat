"""Freeze a cluster-disjoint continuation of the inherited v3 manifest.

The inherited manifest and the original protocol are never overwritten.
No acoustic feature, prediction or classification score is inspected here.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3_continuation"
SEED = 20261023


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(frame, all_rows, previously_scored):
    """Quarantine shared clusters; reserve noise-only clusters without scores."""
    m = frame.copy().reset_index(drop=True)
    m["target"] = m.target.astype(int)
    m["file_id"] = np.arange(len(m))
    m["inherited_role"] = m.role
    locked = m.role.eq("locked_test")
    locked_clusters = set(m.loc[locked, "cluster_merged"])
    shared = ~locked & m.cluster_merged.isin(locked_clusters)
    m.loc[shared, "role"] = "quarantine_holdout_related"
    known_components = set(all_rows.loc[all_rows.path.isin(previously_scored), "component"])
    m["previously_scored_component"] = m.component.isin(known_components)
    # A domain ID alone is not evidence that all these recordings came from UFRGS.
    known_ufrgs = set(all_rows.loc[all_rows.raw_source.eq("ufrgs"), "component"])
    m["provenance_note"] = np.where(m.component.isin(known_ufrgs), "linked_to_known_ufrgs",
                                  np.where(locked, "D3_related_origin_unconfirmed", "inherited_corpus"))
    development = m.role.isin(["train", "noise_train"])
    clusters = m.loc[development].groupby("cluster_merged").agg(
        max_target=("target", "max"), origin=("origin", "first"))
    noise_clusters = clusters[clusters.max_target.eq(0)]
    rng = np.random.default_rng(SEED)
    picked = []
    for origin, sub in noise_clusters.groupby("origin", sort=True):
        candidates = np.sort(sub.index.to_numpy())
        if len(candidates) < 5:
            continue
        count = max(1, int(np.floor(len(candidates) * 0.2)))
        picked.extend(rng.choice(candidates, count, replace=False).tolist())
    m.loc[development & m.cluster_merged.isin(picked), "role"] = "locked_noise"
    assert (m.loc[m.role.eq("locked_noise"), "target"] == 0).all()
    dev = m[m.role.isin(["train", "noise_train"])]
    test = m[m.role.isin(["locked_test", "locked_noise"])]
    for key in ["path", "component", "cluster_merged"]:
        if set(dev[key]) & set(test[key]):
            raise ValueError(f"Development/final-test overlap in {key}")
    if not m.path.is_unique or not m.component.is_unique:
        raise ValueError("Manifest must contain exactly one representative per duplicate component")
    return m


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inherited = ROOT / "results/v3/manifest_v3.csv"
    all_path = ROOT / "results/v3/manifest_v3_all_with_exclusions.csv"
    old_path = ROOT / "reanalysis/outputs/legacy_model_external.csv"
    frame = pd.read_csv(inherited)
    all_rows = pd.read_csv(all_path)
    previously_scored = set(pd.read_csv(old_path).path)
    m = prepare(frame, all_rows, previously_scored)
    manifest = OUT / "manifest.csv"
    if manifest.exists():
        existing = pd.read_csv(manifest)
        if not existing.fillna("").astype(str).equals(m.fillna("").astype(str)):
            raise ValueError("Refusing to replace a different continuation manifest")
    else:
        m.to_csv(manifest, index=False)
    inventory = m.groupby(["role", "origin", "target"]).agg(
        files=("path", "size"), clusters=("cluster_merged", "nunique"), seconds=("duration_s", "sum"))
    inventory.to_csv(OUT / "inventory.csv")
    audit = {
        "seed": SEED,
        "inherited_manifest_sha256": digest(inherited),
        "all_rows_sha256": digest(all_path),
        "manifest_sha256": digest(manifest),
        "n_files": len(m),
        "roles": m.role.value_counts().to_dict(),
        "locked_aedes_clusters": int(m.loc[m.role.eq("locked_test") & m.target.eq(2), "cluster_merged"].nunique()),
        "locked_previously_scored_components": int(m.loc[m.role.eq("locked_test"), "previously_scored_component"].sum()),
        "overlap_development_test_paths_components_clusters": 0,
        "classification_results_inspected": False,
    }
    (OUT / "manifest_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(inventory.round(2).to_string())
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
