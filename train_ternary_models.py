"""
Treinamento e Avaliação Out-Of-Fold da Reforma de Detecção Ternária.
mosquito-wingbeat - 2026-10-14

Classes:
  0: no_mosquito (ruídos ambientais, gravações de fundo, ESC50)
  1: other_mosquito (demais espécies: Anopheles, Culex, Mansonia, Aedes albopictus, etc.)
  2: aedes_aegypti (apenas Aedes aegypti com procedência auditada)

Expurgos aplicados:
  - UFRGS (datasets que não produzem resultado em transferência)
  - unknown_mosquito (rótulos ambíguos)
  - Figshare (atalho positivo com captação restrita)
  - quality_ok == False
  - Features bioacústicas não finitas
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support
)

ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "results/sex_balanced_reliability_2026_10_01/segments.csv"
DATA_FEAT = ROOT / "data/cache/reliability_2026_10_01/features.npy"
OUT_DIR = ROOT / "results/ternary_detection_2026_10_14"

SEED = 20261014
CLASS_NAMES = ["no_mosquito", "other_mosquito", "aedes_aegypti"]


def load_and_filter_data():
    if not DATA_CSV.exists() or not DATA_FEAT.exists():
        raise FileNotFoundError(f"Dados não encontrados em {DATA_CSV} ou {DATA_FEAT}")

    df = pd.read_csv(DATA_CSV, low_memory=False)
    X_raw = np.load(DATA_FEAT, mmap_mode="r")

    if len(df) != len(X_raw):
        raise ValueError(f"Tamanhos divergentes: {len(df)} linhas em CSV vs {len(X_raw)} em features.npy")

    # 1. Filtro de UFRGS
    mask_ufrgs = (
        df["domain"].astype(str).str.lower().eq("ufrgs") |
        df["path"].astype(str).str.contains("ufrgs", case=False, na=False) |
        df["group"].astype(str).str.contains("ufrgs", case=False, na=False)
    )

    # 2. Filtro de Figshare
    mask_figshare = (
        df["domain"].astype(str).str.lower().eq("figshare") |
        df["path"].astype(str).str.contains("figshare", case=False, na=False)
    )

    # 3. Filtro de unknown_mosquito
    mask_unknown = df["label"].astype(str).str.lower().isin(["unknown_mosquito", "unknown", "mosquito"])

    # 4. Qualidade técnica do áudio
    mask_quality = df["quality_ok"].isin([True, "True", "true", 1, "1"])

    # 5. Validação de features finitas
    X_mat = np.array(X_raw, dtype="float32", copy=True)
    mask_finite = np.isfinite(X_mat).all(axis=1)

    # Combinação de filtros
    valid_idx = (~mask_ufrgs) & (~mask_figshare) & (~mask_unknown) & mask_quality & mask_finite
    df_clean = df.loc[valid_idx].copy().reset_index(drop=True)
    X_clean = X_mat[valid_idx.to_numpy()]

    # Mapeamento ternário
    # 0: no_mosquito
    # 2: aedes_aegypti
    # 1: other_mosquito
    labels_lower = df_clean["label"].astype(str).str.lower()
    domains_lower = df_clean["domain"].astype(str).str.lower()

    y = np.full(len(df_clean), -1, dtype=int)
    mask_c0 = labels_lower.eq("noise") | domains_lower.eq("esc50")
    mask_c2 = labels_lower.eq("aedes aegypti")
    mask_c1 = (~mask_c0) & (~mask_c2)

    y[mask_c0] = 0
    y[mask_c1] = 1
    y[mask_c2] = 2

    assert (y >= 0).all(), "Existem instâncias com mapeamento ternário inválido."
    df_clean["ternary_target"] = y
    df_clean["ternary_class_name"] = [CLASS_NAMES[val] for val in y]

    stats = {
        "total_original": len(df),
        "excluded_ufrgs": int(mask_ufrgs.sum()),
        "excluded_figshare": int(mask_figshare.sum()),
        "excluded_unknown": int(mask_unknown.sum()),
        "excluded_quality": int((~mask_quality).sum()),
        "excluded_non_finite": int((~mask_finite).sum()),
        "retained_total": len(df_clean),
        "class_counts": {CLASS_NAMES[i]: int((y == i).sum()) for i in range(3)},
        "group_counts": {CLASS_NAMES[i]: int(df_clean.loc[y == i, "group"].nunique()) for i in range(3)},
    }
    return df_clean, X_clean, y, stats


def compute_sample_weights(df, y, gamma=0.50):
    """
    Renormalização de Classe após Amortecimento por Duração:
      1. u_i = 1.0 / (clip(N_janelas(rec_i), 1, 60) ** 0.50)
      2. S_c = sum_{i in c} u_i para c in {0, 1, 2}
      3. w_i = u_i * (N_total / (3.0 * S_c[y_i]))
    Garante rigorosamente que sum_{i in c} w_i = N_total / 3 para cada classe.
    """
    n_total = len(y)
    rec_window_counts = df["group"].map(df["group"].value_counts()).to_numpy(dtype=np.float64)
    clipped_windows = np.clip(rec_window_counts, 1.0, 60.0)
    u = 1.0 / (clipped_windows ** gamma)

    # S_c = soma dos amortecimentos u_i por classe
    s_c = np.zeros(3, dtype=np.float64)
    for c in range(3):
        s_c[c] = np.sum(u[y == c])

    class_scales = n_total / (3.0 * s_c)
    sample_weights = u * class_scales[y]

    # Validação estrita de garantia de massa por classe
    for c in range(3):
        sum_c = float(np.sum(sample_weights[y == c]))
        expected_c = n_total / 3.0
        assert np.isclose(sum_c, expected_c, rtol=1e-5), (
            f"Renormalização falhou na classe {c}: {sum_c} != {expected_c}"
        )

    return sample_weights.astype(np.float64)


def compute_metrics(y_true, prob, name):
    pred = np.argmax(prob, axis=1)
    acc = float(accuracy_score(y_true, pred))
    b_acc = float(balanced_accuracy_score(y_true, pred))
    cm_raw = confusion_matrix(y_true, pred, labels=[0, 1, 2])
    cm_norm = confusion_matrix(y_true, pred, labels=[0, 1, 2], normalize="true")

    # FPR em ruído puro classificado como Aedes aegypti
    noise_mask = (y_true == 0)
    noise_total = int(noise_mask.sum())
    fpr_noise_to_aedes = float((pred[noise_mask] == 2).sum() / max(1, noise_total))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, pred, labels=[0, 1, 2], zero_division=0
    )

    per_class = {}
    for idx, cname in enumerate(CLASS_NAMES):
        per_class[cname] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx])
        }

    return {
        "model": name,
        "accuracy": acc,
        "balanced_accuracy": b_acc,
        "noise_to_aedes_fpr": fpr_noise_to_aedes,
        "per_class": per_class,
        "confusion_matrix_raw": cm_raw.tolist(),
        "confusion_matrix_norm": cm_norm.tolist()
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, X, y, data_stats = load_and_filter_data()
    groups = df["group"].to_numpy()

    print("=== CORPUS TERNÁRIO DEPOIS DO EXPURGO ===")
    print(json.dumps(data_stats, indent=2))

    # Cálculo dos pesos compostos com renormalização de classe (gamma = 0.50)
    sample_weights = compute_sample_weights(df, y, gamma=0.50)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    oof_lr = np.zeros((len(df), 3), dtype=np.float64)
    oof_hgb = np.zeros((len(df), 3), dtype=np.float64)
    fold_assignments = np.zeros(len(df), dtype=int)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        assert len(train_groups & val_groups) == 0, f"Vazamento entre dobras no fold {fold}!"

        fold_assignments[val_idx] = fold
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        sw_train = sample_weights[train_idx]

        # 1. Logistic Regression com StandardScaler ajustado exclusivamente na dobra de treino
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        lr = LogisticRegression(
            class_weight=None,
            max_iter=2000,
            C=0.1,
            random_state=SEED
        )
        lr.fit(X_train_scaled, y_train, sample_weight=sw_train)
        oof_lr[val_idx] = lr.predict_proba(X_val_scaled)

        # 2. HistGradientBoostingClassifier multiclasse treinado com sample_weight
        hgb = HistGradientBoostingClassifier(
            max_iter=200,
            l2_regularization=3.0,
            learning_rate=0.05,
            random_state=SEED
        )
        hgb.fit(X_train, y_train, sample_weight=sw_train)
        oof_hgb[val_idx] = hgb.predict_proba(X_val)

        print(f"Fold {fold + 1}/5 concluído.")

    # 3. Soft Voting Ensemble
    oof_ens = (oof_lr + oof_hgb) / 2.0

    # Normalização estrita para garantir soma exatamente 1.0 por linha
    oof_lr = oof_lr / oof_lr.sum(axis=1, keepdims=True)
    oof_hgb = oof_hgb / oof_hgb.sum(axis=1, keepdims=True)
    oof_ens = oof_ens / oof_ens.sum(axis=1, keepdims=True)

    # Validação estrita de probabilidades
    for p, name in [(oof_lr, "LogisticRegression"), (oof_hgb, "HistGradientBoosting"), (oof_ens, "Ensemble")]:
        assert np.isfinite(p).all(), f"Probabilidades com NaN/Inf em {name}"
        assert (p >= 0).all(), f"Probabilidades negativas em {name}"
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5), f"Soma != 1 em {name}"

    # Cálculo das Métricas
    metrics_lr = compute_metrics(y, oof_lr, "logistic_regression")
    metrics_hgb = compute_metrics(y, oof_hgb, "hist_gradient_boosting")
    metrics_ens = compute_metrics(y, oof_ens, "ensemble")

    metrics_bundle = {
        "protocol": {
            "n_splits": 5,
            "seed": SEED,
            "validation": "StratifiedGroupKFold",
            "group_leakage": 0,
            "sample_weight": {
                "method": "duration_damping_with_class_renormalization",
                "formula": "w_i = u_i * (N_total / (3 * S_c)), u_i = clip(N_janelas(rec_i), 1, 60)^(-0.50)",
                "gamma": 0.50,
                "clip_min": 1,
                "clip_max": 60
            }
        },
        "dataset_statistics": data_stats,
        "models": {
            "logistic_regression": {k: v for k, v in metrics_lr.items() if not k.startswith("confusion_matrix")},
            "hist_gradient_boosting": {k: v for k, v in metrics_hgb.items() if not k.startswith("confusion_matrix")},
            "ensemble": {k: v for k, v in metrics_ens.items() if not k.startswith("confusion_matrix")}
        }
    }

    confusion_matrices = {
        "class_labels": CLASS_NAMES,
        "logistic_regression": {
            "raw": metrics_lr["confusion_matrix_raw"],
            "normalized": metrics_lr["confusion_matrix_norm"]
        },
        "hist_gradient_boosting": {
            "raw": metrics_hgb["confusion_matrix_raw"],
            "normalized": metrics_hgb["confusion_matrix_norm"]
        },
        "ensemble": {
            "raw": metrics_ens["confusion_matrix_raw"],
            "normalized": metrics_ens["confusion_matrix_norm"]
        }
    }

    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics_bundle, indent=2, ensure_ascii=False))
    (OUT_DIR / "confusion_matrix.json").write_text(json.dumps(confusion_matrices, indent=2, ensure_ascii=False))

    pred_df = pd.DataFrame({
        "path": df["path"],
        "group": df["group"],
        "fold": fold_assignments,
        "true_label": y,
        "true_class_name": df["ternary_class_name"],
        "lr_prob_0": oof_lr[:, 0],
        "lr_prob_1": oof_lr[:, 1],
        "lr_prob_2": oof_lr[:, 2],
        "lr_pred": np.argmax(oof_lr, axis=1),
        "hgb_prob_0": oof_hgb[:, 0],
        "hgb_prob_1": oof_hgb[:, 1],
        "hgb_prob_2": oof_hgb[:, 2],
        "hgb_pred": np.argmax(oof_hgb, axis=1),
        "ensemble_prob_0": oof_ens[:, 0],
        "ensemble_prob_1": oof_ens[:, 1],
        "ensemble_prob_2": oof_ens[:, 2],
        "ensemble_pred": np.argmax(oof_ens, axis=1)
    })
    pred_df.to_csv(OUT_DIR / "predictions_oof.csv", index=False)

    print("\n=== RESULTADOS OOF DETECÇÃO TERNÁRIA ===")
    for m in [metrics_lr, metrics_hgb, metrics_ens]:
        print(f"\n--- Modelo: {m['model']} ---")
        print(f"Acurácia Global: {m['accuracy']:.4f}")
        print(f"Acurácia Balanceada: {m['balanced_accuracy']:.4f}")
        print(f"FPR Ruído -> Aedes: {m['noise_to_aedes_fpr']:.4f}")
        for cname in CLASS_NAMES:
            c = m["per_class"][cname]
            print(f"  {cname:15s} | Prec: {c['precision']:.4f} | Rec: {c['recall']:.4f} | F1: {c['f1']:.4f}")


if __name__ == "__main__":
    import sys
    if "--nested" in sys.argv:
        from train_ternary_nested_models import run_nested_training
        run_nested_training()
    else:
        main()
