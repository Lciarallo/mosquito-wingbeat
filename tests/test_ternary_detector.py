"""
Suíte de testes para validação da reforma de detecção ternária.
Garante:
  1. Classes estritamente em {0, 1, 2}.
  2. Particionamento 100% disjunto por grupo (zero leakage).
  3. Probabilidades válidas, finitas e com soma 1.
  4. Consistência matemática e estrutural de todos os artefatos gerados.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results/ternary_detection_2026_10_14"
CSV_PATH = RESULTS_DIR / "predictions_oof.csv"
METRICS_PATH = RESULTS_DIR / "metrics.json"
CONF_PATH = RESULTS_DIR / "confusion_matrix.json"


@pytest.fixture(scope="module")
def artifacts():
    assert RESULTS_DIR.exists(), f"Diretório {RESULTS_DIR} não existe."
    assert CSV_PATH.exists(), f"Arquivo {CSV_PATH} não existe."
    assert METRICS_PATH.exists(), f"Arquivo {METRICS_PATH} não existe."
    assert CONF_PATH.exists(), f"Arquivo {CONF_PATH} não existe."

    df = pd.read_csv(CSV_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    conf = json.loads(CONF_PATH.read_text(encoding="utf-8"))
    return df, metrics, conf


def test_classes_strictly_ternary(artifacts):
    df, metrics, _ = artifacts
    unique_true = set(df["true_label"].unique())
    assert unique_true == {0, 1, 2}, f"Classes verdadeiras devem ser exatamente {{0, 1, 2}}, obtido {unique_true}"

    for prefix in ["lr", "hgb", "ensemble"]:
        unique_pred = set(df[f"{prefix}_pred"].unique())
        assert unique_pred.issubset({0, 1, 2}), f"Classes preditas para {prefix} fora de {{0, 1, 2}}: {unique_pred}"

    # Verifica consistência com as contagens salvas
    class_counts = metrics["dataset_statistics"]["class_counts"]
    for idx, cname in enumerate(["no_mosquito", "other_mosquito", "aedes_aegypti"]):
        assert (df["true_label"] == idx).sum() == class_counts[cname]


def test_group_kfold_strictly_disjoint(artifacts):
    df, _, _ = artifacts
    groups = df["group"].to_numpy()
    folds = df["fold"].to_numpy()

    # Verifica que cada grupo pertence a uma e apenas uma dobra
    for group_name, part in df.groupby("group"):
        unique_folds = part["fold"].unique()
        assert len(unique_folds) == 1, f"Grupo {group_name} vazou em múltiplas dobras: {unique_folds}"

    # Verifica par a par entre dobras
    for f in range(5):
        val_groups = set(groups[folds == f])
        train_groups = set(groups[folds != f])
        intersection = val_groups & train_groups
        assert len(intersection) == 0, f"Vazamento detectado no fold {f}: {intersection}"


def test_probabilities_valid_sum_to_one(artifacts):
    df, _, _ = artifacts
    for prefix in ["lr", "hgb", "ensemble"]:
        cols = [f"{prefix}_prob_{i}" for i in range(3)]
        probs = df[cols].to_numpy()

        assert np.isfinite(probs).all(), f"Probabilidades em {prefix} contêm NaN ou Inf"
        assert (probs >= 0.0).all(), f"Probabilidades em {prefix} são negativas"
        assert (probs <= 1.0).all(), f"Probabilidades em {prefix} são maiores que 1.0"

        row_sums = probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), f"Soma das probabilidades em {prefix} difere de 1.0"


def test_saved_artifacts_consistency(artifacts):
    df, metrics, conf = artifacts
    n_samples = len(df)
    assert metrics["dataset_statistics"]["retained_total"] == n_samples

    for model_name, prefix in [
        ("logistic_regression", "lr"),
        ("hist_gradient_boosting", "hgb"),
        ("ensemble", "ensemble")
    ]:
        model_metrics = metrics["models"][model_name]
        raw_cm = np.array(conf[model_name]["raw"])
        norm_cm = np.array(conf[model_name]["normalized"])

        assert raw_cm.shape == (3, 3)
        assert norm_cm.shape == (3, 3)
        assert raw_cm.sum() == n_samples

        # Matriz normalizada deve somar 1 por linha (sensibilidade por classe)
        assert np.allclose(norm_cm.sum(axis=1), 1.0, atol=1e-5)

        # Balanced accuracy é a média da diagonal da matriz de confusão normalizada
        expected_b_acc = float(np.diag(norm_cm).mean())
        assert abs(model_metrics["balanced_accuracy"] - expected_b_acc) < 1e-4

        # FPR de ruído para Aedes deve coincidir com norm_cm[0, 2]
        expected_noise_fpr = float(norm_cm[0, 2])
        assert abs(model_metrics["noise_to_aedes_fpr"] - expected_noise_fpr) < 1e-4


def test_event_level_metrics_consistency():
    event_json_path = RESULTS_DIR / "event_level_metrics.json"
    assert event_json_path.exists(), f"{event_json_path} não existe."
    data = json.loads(event_json_path.read_text(encoding="utf-8"))

    assert data["event_summary"]["total_recordings"] == 1504
    for mname in ["logistic_regression_event", "hist_gradient_boosting_event", "ensemble_event", "balanced_blend_event"]:
        m = data["models"][mname]
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["balanced_accuracy"] <= 1.0
        assert 0.0 <= m["aedes_recall"] <= 1.0
        assert 0.0 <= m["noise_to_aedes_fpr"] <= 0.05

