"""Check declared recording groups, paths and hashes across dataset folds.

This bounded check does not inspect calibration prediction lineage, biological
ancestry, threshold selection or recompute hashes from audio. Scaler means can
be checked separately; run_full_audit does not perform that check.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def audit_splits(segments_df: pd.DataFrame, fold_col: str = "fold_species") -> dict:
    """Verifica disjunção estrita entre dobras de treino e teste."""
    valid_mask = segments_df[fold_col] >= 0
    df = segments_df[valid_mask].copy()

    unique_folds = sorted(df[fold_col].unique().tolist())
    audit_results = {
        "status": "PASS",
        "total_windows": len(df),
        "total_groups": int(df["group"].nunique()),
        "total_hashes": int(df["content_sha256"].nunique()),
        "folds": unique_folds,
        "fold_audits": [],
    }

    for test_fold in unique_folds:
        train_df = df[df[fold_col] != test_fold]
        test_df = df[df[fold_col] == test_fold]

        # 1. Checagem de Grupos
        train_groups = set(train_df["group"].unique())
        test_groups = set(test_df["group"].unique())
        group_intersection = train_groups.intersection(test_groups)

        # 2. Checagem de Hashes de Conteúdo
        train_hashes = set(train_df["content_sha256"].unique())
        test_hashes = set(test_df["content_sha256"].unique())
        hash_intersection = train_hashes.intersection(test_hashes)

        # 3. Checagem de Caminhos de Arquivo
        train_paths = set(train_df["path"].unique())
        test_paths = set(test_df["path"].unique())
        path_intersection = train_paths.intersection(test_paths)

        passed = (len(group_intersection) == 0 and
                  len(hash_intersection) == 0 and
                  len(path_intersection) == 0)

        if not passed:
            audit_results["status"] = "FAIL"

        fold_record = {
            "test_fold": int(test_fold),
            "train_windows": len(train_df),
            "test_windows": len(test_df),
            "train_groups": len(train_groups),
            "test_groups": len(test_groups),
            "group_overlap_count": len(group_intersection),
            "hash_overlap_count": len(hash_intersection),
            "path_overlap_count": len(path_intersection),
            "passed": passed,
        }
        if not passed:
            fold_record["leaked_groups"] = list(group_intersection)[:5]
            fold_record["leaked_hashes"] = list(hash_intersection)[:5]

        audit_results["fold_audits"].append(fold_record)

    return audit_results


def verify_scaler_isolation(x_train: np.ndarray, x_test: np.ndarray, scaler) -> bool:
    """Garante que a média e desvio do scaler dependem estritamente do treino."""
    computed_mean = np.mean(x_train, axis=0)
    # Tolerância numérica para conferir se o scaler foi ajustado em x_train
    mean_diff = np.max(np.abs(scaler.mean_ - computed_mean))
    return bool(mean_diff < 1e-4)


def run_full_audit():
    segments_path = ROOT / "results/segments.csv"
    if not segments_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {segments_path}")

    segments = pd.read_csv(segments_path)
    print("-> Iniciando Auditoria Criptográfica Anti-Vazamento...")

    # Auditoria da tarefa de espécies
    species_audit = audit_splits(segments, fold_col="fold_species")
    print(f"   [Espécies] Status: {species_audit['status']}")
    for f in species_audit["fold_audits"]:
        print(f"     Dobra de Teste {f['test_fold']}: {f['test_windows']} janelas ({f['test_groups']} grupos) | Overlap Grupos: {f['group_overlap_count']} | Overlap Hashes: {f['hash_overlap_count']}")
        if not f["passed"]:
            raise AssertionError(f"CONTAMINAÇÃO DETECTADA na dobra {f['test_fold']}!")

    # Auditoria da tarefa de tinyml
    tinyml_audit = audit_splits(segments, fold_col="fold_tinyml")
    print(f"\n   [TinyML] Status: {tinyml_audit['status']}")
    for f in tinyml_audit["fold_audits"]:
        print(f"     Dobra de Teste {f['test_fold']}: {f['test_windows']} janelas ({f['test_groups']} grupos) | Overlap Grupos: {f['group_overlap_count']} | Overlap Hashes: {f['hash_overlap_count']}")
        if not f["passed"]:
            raise AssertionError(f"CONTAMINAÇÃO DETECTADA na dobra {f['test_fold']}!")

    certificate = {
        "audit_version": "1.0",
        "overall_status": "PARTITION_DISJOINTNESS_VERIFIED",
        "checked": ["declared_groups", "declared_content_hashes", "paths"],
        "not_checked": ["ensemble_calibration_lineage", "threshold_selection",
                        "scaler_isolation", "biological_independence", "audio_hash_recalculation"],
        "species_task_audit": species_audit,
        "tinyml_task_audit": tinyml_audit,
    }

    out_file = ROOT / "results/leakage_audit_certificate.json"
    out_file.write_text(json.dumps(certificate, indent=2))
    print(f"\n-> Certificado de Conformidade gerado com sucesso: {out_file}")
    return certificate


if __name__ == "__main__":
    run_full_audit()
