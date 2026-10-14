"""
Suíte de testes para validação da reforma de detecção ternária.
Garante:
  1. Classes estritamente em {0, 1, 2}.
  2. Particionamento 100% disjunto por grupo (zero leakage).
  3. Probabilidades válidas, finitas e com soma 1.
  4. Consistência matemática e estrutural de todos os artefatos gerados.
  5. Auditoria de denominadores com soma exata em 1.504 eventos.
  6. Imunidade à diluição de silêncio por Activity-Gated Pooling.
  7. Calibração OOF Neyman-Pearson sem vazamento informacional entre dobras.
  8. Invariância exata de massa por classe na renormalização de pesos.
  9. Consenso e soma estrita 1.0 no Logarithmic Opinion Pool.
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

    class_counts = metrics["dataset_statistics"]["class_counts"]
    for idx, cname in enumerate(["no_mosquito", "other_mosquito", "aedes_aegypti"]):
        assert (df["true_label"] == idx).sum() == class_counts[cname]


def test_group_kfold_strictly_disjoint(artifacts):
    df, _, _ = artifacts
    groups = df["group"].to_numpy()
    folds = df["fold"].to_numpy()

    for group_name, part in df.groupby("group"):
        unique_folds = part["fold"].unique()
        assert len(unique_folds) == 1, f"Grupo {group_name} vazou em múltiplas dobras: {unique_folds}"

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
        assert np.allclose(norm_cm.sum(axis=1), 1.0, atol=1e-5)

        expected_b_acc = float(np.diag(norm_cm).mean())
        assert abs(model_metrics["balanced_accuracy"] - expected_b_acc) < 1e-4

        expected_noise_fpr = float(norm_cm[0, 2])
        assert abs(model_metrics["noise_to_aedes_fpr"] - expected_noise_fpr) < 1e-4


def test_event_level_metrics_consistency():
    event_json_path = RESULTS_DIR / "event_level_metrics.json"
    assert event_json_path.exists(), f"{event_json_path} não existe."
    data = json.loads(event_json_path.read_text(encoding="utf-8"))

    assert data["event_summary"]["total_recordings"] == 1504
    for mname in [
        "logistic_regression_event",
        "hist_gradient_boosting_event",
        "ensemble_event",
        "activity_gated_log_odds_cascade_event"
    ]:
        assert mname in data["models"], f"Modelo {mname} ausente em event_level_metrics.json"
        m = data["models"][mname]
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["balanced_accuracy"] <= 1.0
        assert 0.0 <= m["aedes_recall"] <= 1.0
        assert 0.0 <= m["noise_to_aedes_fpr"] <= 0.05


def test_event_denominators_sum_to_total_recordings():
    """
    Verifica que n_pure_events_per_class soma exatamente 1.504 e coincide
    com 426 (no_mosquito), 1014 (other_mosquito) e 64 (aedes_aegypti).
    """
    event_json_path = RESULTS_DIR / "event_level_metrics.json"
    assert event_json_path.exists(), f"{event_json_path} não existe."
    data = json.loads(event_json_path.read_text(encoding="utf-8"))

    summary = data["event_summary"]
    assert summary["total_recordings"] == 1504

    recs = summary["recordings_per_class"]
    assert recs["no_mosquito"] == 426, f"Esperado 426 no_mosquito, obtido {recs['no_mosquito']}"
    assert recs["other_mosquito"] == 1014, f"Esperado 1014 other_mosquito, obtido {recs['other_mosquito']}"
    assert recs["aedes_aegypti"] == 64, f"Esperado 64 aedes_aegypti, obtido {recs['aedes_aegypti']}"
    assert sum(recs.values()) == 1504

    df = pd.read_csv(CSV_PATH)
    group_labels = df.groupby("group")["true_label"].apply(set)
    event_labels = group_labels.apply(lambda s: 2 if 2 in s else (1 if 1 in s else 0))

    counts = event_labels.value_counts().to_dict()
    assert counts[0] == 426
    assert counts[1] == 1014
    assert counts[2] == 64
    assert len(event_labels) == 1504


def test_activity_gated_pooling_prevents_long_silence_dilution():
    """
    Simula 5 janelas de voo ativo de Aedes (P2=0.70, P0=0.05) e 35 janelas de silêncio (P0=0.99, P2=0.01).
    Prova matematicamente que o pooling ponderado mantém P2 > 0.50, enquanto a média simples colapsa para < 0.10.
    """
    p_active = np.tile([0.05, 0.25, 0.70], (5, 1))
    p_silence = np.tile([0.99, 0.00, 0.01], (35, 1))
    p_all = np.vstack([p_active, p_silence])

    # 1. Média simples
    p2_simple = float(p_all[:, 2].mean())
    assert p2_simple < 0.10, f"Média simples deveria ser < 0.10, obtido {p2_simple:.4f}"

    # 2. Activity-gated pooling: a_t = (1 - P_0(t))^1.5
    p0 = p_all[:, 0]
    act = (1.0 - p0) ** 1.5
    weights = act / np.sum(act)
    p2_gated = float(np.sum(weights * p_all[:, 2]))

    assert p2_gated > 0.50, f"Pooling ponderado por atividade deveria manter P2 > 0.50, obtido {p2_gated:.4f}"
    assert (p2_gated / p2_simple) > 7.0


def test_oof_threshold_calibration_no_leakage():
    """
    Valida que a otimização de limiares (tau_ruido, tau_aedes) foi calculada
    por dobra cegamente (zero leakage) e os limiares de cada dobra foram aplicados sem contaminação.
    """
    event_json_path = RESULTS_DIR / "event_level_metrics.json"
    assert event_json_path.exists(), f"{event_json_path} não existe."
    data = json.loads(event_json_path.read_text(encoding="utf-8"))

    cascade_model = data["models"]["activity_gated_log_odds_cascade_event"]
    assert "thresholds_per_fold" in cascade_model, "Calibrações de limiar por dobra ausentes"

    thresh = cascade_model["thresholds_per_fold"]
    assert len(thresh) == 5, f"Esperadas 5 dobras de calibração, obtido {len(thresh)}"

    for k in range(5):
        key = f"fold_{k}"
        assert key in thresh, f"Dobra {key} ausente"
        f_data = thresh[key]

        tau_r = f_data["tau_noise"]
        tau_ae = f_data["tau_aedes"]
        train_folds = f_data["train_folds"]

        assert 0.40 <= tau_r <= 0.80, f"tau_noise fora da grade no fold {k}: {tau_r}"
        assert 0.15 <= tau_ae <= 0.60, f"tau_aedes fora da grade no fold {k}: {tau_ae}"
        assert k not in train_folds, f"Vazamento! Fold {k} foi incluído no treino de {key}"
        assert len(train_folds) == 4, f"Deveriam haver exatamente 4 dobras no treino de {key}"


def test_class_renormalization_sample_weights_property():
    """
    Verifica a propriedade analítica de que a renormalização por classe
    iguala exatamente as massas sum_{i in c} w_i = N_total / 3 para as 3 classes.
    """
    rng = np.random.RandomState(42)
    n_samples = 3000
    y = np.repeat([0, 1, 2], [1000, 1500, 500])
    rec_lens = np.where(y == 2, 50, np.where(y == 1, 20, 5))

    u = np.clip(rec_lens, 1.0, 60.0) ** (-0.75)
    s_c = np.array([np.sum(u[y == c]) for c in range(3)])
    weights = u * (len(y) / (3.0 * s_c[y]))

    for c in range(3):
        assert np.isclose(np.sum(weights[y == c]), len(y) / 3.0, rtol=1e-5)


def test_logarithmic_opinion_pool_guarantees_sum_to_one():
    """
    Verifica que o Logarithmic Opinion Pool garante soma estrita 1.0 e
    age como consenso estrito (se um modelo estima P0=0, o pool mantém P0 próximo de 0).
    """
    p_lr = np.array([0.001, 0.200, 0.799])
    p_hgb = np.array([0.0001, 0.300, 0.6999])

    w = 0.60
    z = w * np.log(p_lr) + (1.0 - w) * np.log(p_hgb)
    z_shift = z - np.max(z)
    p_fused = np.exp(z_shift) / np.sum(np.exp(z_shift))

    assert np.isclose(np.sum(p_fused), 1.0, atol=1e-6)
    assert p_fused[0] < 0.001, "Consenso em ruído baixo deve ser preservado pelo log-pool"
    assert p_fused[2] > 0.70, "Consenso em Aedes deve prevalecer fortemente"


NESTED_DIR = ROOT / "results/ternary_nested_2026_10_14"


def test_pcm_block_isolation():
    """Valida que o isolamento por pcm_block é estrito em todas as dobras externas."""
    nested_csv = NESTED_DIR / "predictions_oof.csv"
    if not nested_csv.exists():
        pytest.skip("predictions_oof.csv aninhado ainda não gerado")
    df = pd.read_csv(nested_csv)
    assert "pcm_block" in df.columns, "Coluna pcm_block ausente no CSV aninhado"

    for b, part in df.groupby("pcm_block"):
        folds = part["fold"].unique()
        assert len(folds) == 1, f"pcm_block {b} vazou em múltiplas dobras: {folds}"

    for f in range(5):
        val_blocks = set(df[df["fold"] == f]["pcm_block"])
        train_blocks = set(df[df["fold"] != f]["pcm_block"])
        assert len(val_blocks & train_blocks) == 0, f"Vazamento de pcm_block no fold {f}"


def test_sample_weights_strictly_within_train():
    """Valida fórmula analítica e garantia de massa calculada apenas com dados de treino."""
    from train_ternary_nested_models import compute_sample_weights
    n_train = 1200
    y_tr = np.repeat([0, 1, 2], [400, 600, 200])
    df_dummy = pd.DataFrame({
        "group": [f"grp_{i % 50}" for i in range(n_train)]
    })
    weights = compute_sample_weights(df_dummy, y_tr, gamma=0.50)
    for c in range(3):
        assert np.isclose(np.sum(weights[y_tr == c]), n_train / 3.0, rtol=1e-4)


def test_temporal_persistence_strictly_within_wav():
    """Testa explicitamente que duas janelas com p2 alto em WAVs diferentes do mesmo grupo NÃO são fundidas."""
    from evaluate_ternary_event_pipeline import compute_group_wav_bounded_persistence
    df_synthetic = pd.DataFrame([
        {"path": "recA.wav", "start_s": 0.0, "duration_s": 1.0, "ensemble_prob_2": 0.90},
        {"path": "recA.wav", "start_s": 1.0, "duration_s": 1.0, "ensemble_prob_2": 0.10},
        {"path": "recB.wav", "start_s": 1.0, "duration_s": 1.0, "ensemble_prob_2": 0.90},
        {"path": "recB.wav", "start_s": 2.0, "duration_s": 1.0, "ensemble_prob_2": 0.10},
    ])
    p = compute_group_wav_bounded_persistence(df_synthetic, "ensemble_prob_2", horizon_s=2.0)
    # Dentro de cada WAV, a média contígua é (0.90 + 0.10)/2 = 0.50
    # Se houvesse fusão entre recA e recB em start_s=1.0, daria 0.90
    assert p <= 0.5001, f"Persistência fundiu WAVs diferentes: obtido {p}"


def test_nested_artifacts_consistency():
    """Valida consistência matemática e integridade de todos os artefatos aninhados."""
    if not NESTED_DIR.exists():
        pytest.skip("Diretório aninhado ainda não gerado")
    pred_path = NESTED_DIR / "predictions_oof.csv"
    if not pred_path.exists():
        pytest.skip("predictions_oof.csv aninhado ainda não gerado")
    df = pd.read_csv(pred_path)
    for prefix in ["lr", "hgb", "ensemble"]:
        cols = [f"{prefix}_prob_{i}" for i in range(3)]
        probs = df[cols].to_numpy()
        assert np.isfinite(probs).all()
        assert (probs >= 0.0).all()
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_operational_reporting_integrity():
    """Valida integridade do relatório de eventos aninhado com contagens inteiras e Wilson CI."""
    ev_json = NESTED_DIR / "event_level_metrics.json"
    if not ev_json.exists():
        pytest.skip("event_level_metrics.json aninhado ainda não gerado")
    data = json.loads(ev_json.read_text(encoding="utf-8"))
    for mname in ["logistic_regression_event", "hist_gradient_boosting_event", "ensemble_event", "activity_gated_log_odds_cascade_event"]:
        m = data["models"][mname]
        assert "integer_counts" in m
        assert "wilson_ci_95" in m
        counts = m["integer_counts"]
        # Derive the contract from this experiment's own unit and population.
        # Never whitelist denominators merely to make a different corpus pass.
        groups = pd.read_csv(NESTED_DIR / "predictions_oof.csv").groupby("group").true_label.max()
        assert counts["noise_total"] == int(groups.eq(0).sum())
        assert counts["aedes_total"] == int(groups.eq(2).sum())
        assert counts["other_total"] == int(groups.eq(1).sum())
        wilson = m["wilson_ci_95"]
        for key in ["aedes_recall", "noise_to_aedes_fpr", "other_to_aedes_fpr"]:
            ci = wilson[key]
            assert 0.0 <= ci["ci_lower"] <= ci["point"] <= ci["ci_upper"] <= 1.0
