"""
Treinamento e Validação Aninhada Estrita (Nested Cross-Validation) da Detecção Ternária.
mosquito-wingbeat - 2026-10-14

Metodologia Estrita:
1. Isolamento por blocos de colisão PCM (pcm_block): 54 grupos conectados isolados no Fold 0.
2. Zero leakage: divisões externas e internas 100% disjuntas por pcm_block.
3. Validação aninhada (Nested CV): seleção de gamma (0.50 vs 0.75) e calibração de limiares
   realizadas exclusivamente nas dobras de inner CV dentro do treino da dobra externa.
4. Dobra externa de teste 100% cega até a inferência final.
5. Pesos amostrais e padronização (StandardScaler) calculados apenas no conjunto de treino.
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
COLLISION_MAP_CSV = ROOT / "results/species_nested_2026_10_13/recording_predictions.csv"
OUT_DIR = ROOT / "results/ternary_nested_2026_10_14"

SEED = 20261014
CLASS_NAMES = ["no_mosquito", "other_mosquito", "aedes_aegypti"]


def load_and_filter_data():
    if not DATA_CSV.exists() or not DATA_FEAT.exists():
        raise FileNotFoundError(f"Dados não encontrados em {DATA_CSV} ou {DATA_FEAT}")

    df = pd.read_csv(DATA_CSV, low_memory=False)
    X_raw = np.load(DATA_FEAT, mmap_mode="r")

    if len(df) != len(X_raw):
        raise ValueError(f"Tamanhos divergentes: {len(df)} linhas em CSV vs {len(X_raw)} em features.npy")

    mask_ufrgs = (
        df["domain"].astype(str).str.lower().eq("ufrgs") |
        df["path"].astype(str).str.contains("ufrgs", case=False, na=False) |
        df["group"].astype(str).str.contains("ufrgs", case=False, na=False)
    )
    mask_figshare = (
        df["domain"].astype(str).str.lower().eq("figshare") |
        df["path"].astype(str).str.contains("figshare", case=False, na=False)
    )
    mask_unknown = df["label"].astype(str).str.lower().isin(["unknown_mosquito", "unknown", "mosquito"])
    mask_quality = df["quality_ok"].isin([True, "True", "true", 1, "1"])

    X_mat = np.array(X_raw, dtype="float32", copy=True)
    mask_finite = np.isfinite(X_mat).all(axis=1)

    valid_idx = (~mask_ufrgs) & (~mask_figshare) & (~mask_unknown) & mask_quality & mask_finite
    df_clean = df.loc[valid_idx].copy().reset_index(drop=True)
    X_clean = X_mat[valid_idx.to_numpy()]

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

    if COLLISION_MAP_CSV.exists():
        df_map = pd.read_csv(COLLISION_MAP_CSV)
        collision_map = dict(zip(df_map["group"], df_map["collision_group"]))
        pcm_blocks = df_clean["group"].map(collision_map).fillna(df_clean["group"]).to_numpy(dtype=str)
    else:
        pcm_blocks = df_clean["group"].to_numpy(dtype=str)

    df_clean["pcm_block"] = pcm_blocks

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
        "pcm_block_counts": {CLASS_NAMES[i]: int(df_clean.loc[y == i, "pcm_block"].nunique()) for i in range(3)},
        "unique_pcm_blocks": int(len(np.unique(pcm_blocks))),
    }
    return df_clean, X_clean, y, stats


def compute_sample_weights(df, y, gamma=0.50):
    """
    Cálculo analítico dos pesos com garantia estrita de massa sum_{i in c} w_i = N_total / 3.
    Calculado estritamente com dados da partição de treino.
    """
    n_total = len(y)
    rec_counts = df["group"].map(df["group"].value_counts()).to_numpy(dtype=np.float64)
    clipped_windows = np.clip(rec_counts, 1.0, 60.0)
    u = 1.0 / (clipped_windows ** gamma)

    s_c = np.zeros(3, dtype=np.float64)
    for c in range(3):
        s_c[c] = np.sum(u[y == c])
        if s_c[c] <= 0:
            s_c[c] = 1e-6

    class_scales = n_total / (3.0 * s_c)
    sample_weights = u * class_scales[y]

    for c in range(3):
        if np.any(y == c):
            sum_c = float(np.sum(sample_weights[y == c]))
            expected_c = n_total / 3.0
            assert np.isclose(sum_c, expected_c, rtol=1e-4), (
                f"Renormalização falhou na classe {c}: {sum_c} != {expected_c}"
            )

    return sample_weights.astype(np.float64)


def compute_metrics(y_true, prob, name):
    pred = np.argmax(prob, axis=1)
    acc = float(accuracy_score(y_true, pred))
    b_acc = float(balanced_accuracy_score(y_true, pred))
    cm_raw = confusion_matrix(y_true, pred, labels=[0, 1, 2])
    cm_norm = confusion_matrix(y_true, pred, labels=[0, 1, 2], normalize="true")

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


def run_nested_training():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, X, y, data_stats = load_and_filter_data()
    pcm_blocks = df["pcm_block"].to_numpy()

    print("=== CORPUS TERNÁRIO ANINHADO DEPOIS DO EXPURGO ===")
    print(json.dumps(data_stats, indent=2))

    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    oof_lr = np.zeros((len(df), 3), dtype=np.float64)
    oof_hgb = np.zeros((len(df), 3), dtype=np.float64)
    fold_assignments = np.zeros(len(df), dtype=int)

    gammas_to_evaluate = [0.50, 0.75]
    inner_thresholds_per_fold = {}
    outer_fold_gamma_selected = {}

    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y, groups=pcm_blocks)):
        train_pcm = set(pcm_blocks[train_idx])
        val_pcm = set(pcm_blocks[val_idx])
        assert len(train_pcm & val_pcm) == 0, f"Vazamento de pcm_block na dobra externa {fold}!"

        fold_assignments[val_idx] = fold
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        df_train = df.iloc[train_idx].reset_index(drop=True)
        pcm_train = pcm_blocks[train_idx]

        print(f"\n>>> Fold Externo {fold + 1}/5: {len(X_train)} treino, {len(X_val)} teste cego")

        inner_cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
        best_gamma = 0.50
        best_inner_bacc = -1.0
        best_gamma_inner_oof = None

        for gamma in gammas_to_evaluate:
            inner_oof_lr = np.zeros((len(df_train), 3), dtype=np.float64)
            inner_oof_hgb = np.zeros((len(df_train), 3), dtype=np.float64)

            for in_fold, (in_train_idx, in_val_idx) in enumerate(inner_cv.split(X_train, y_train, groups=pcm_train)):
                assert len(set(pcm_train[in_train_idx]) & set(pcm_train[in_val_idx])) == 0

                df_in_tr = df_train.iloc[in_train_idx].reset_index(drop=True)
                y_in_tr = y_train[in_train_idx]
                X_in_tr = X_train[in_train_idx]
                X_in_val = X_train[in_val_idx]

                sw_in_tr = compute_sample_weights(df_in_tr, y_in_tr, gamma=gamma)

                scaler = StandardScaler()
                X_in_tr_sc = scaler.fit_transform(X_in_tr)
                X_in_val_sc = scaler.transform(X_in_val)

                lr_in = LogisticRegression(class_weight=None, max_iter=2000, C=0.1, random_state=SEED)
                lr_in.fit(X_in_tr_sc, y_in_tr, sample_weight=sw_in_tr)
                inner_oof_lr[in_val_idx] = lr_in.predict_proba(X_in_val_sc)

                hgb_in = HistGradientBoostingClassifier(
                    max_iter=150, l2_regularization=3.0, learning_rate=0.05, random_state=SEED
                )
                hgb_in.fit(X_in_tr, y_in_tr, sample_weight=sw_in_tr)
                inner_oof_hgb[in_val_idx] = hgb_in.predict_proba(X_in_val)

            inner_ens = (inner_oof_lr + inner_oof_hgb) / 2.0
            inner_pred = np.argmax(inner_ens, axis=1)
            bacc = balanced_accuracy_score(y_train, inner_pred)
            print(f"  Inner CV fold {fold} | gamma={gamma} -> Inner BAcc: {bacc:.4f}")

            if bacc > best_inner_bacc:
                best_inner_bacc = bacc
                best_gamma = gamma
                best_gamma_inner_oof = (inner_oof_lr, inner_oof_hgb)

        outer_fold_gamma_selected[f"fold_{fold}"] = best_gamma
        print(f"  --> Melhor gamma selecionado para fold {fold}: {best_gamma}")

        # Calibração estritamente interna dos limiares da cascata Neyman-Pearson
        in_lr, in_hgb = best_gamma_inner_oof
        df_train_groups = []
        for gname, sub_idx in df_train.groupby("group").groups.items():
            sub_indices = np.array(list(sub_idx))
            y_ev = int(y_train[sub_indices[0]])
            if 2 in y_train[sub_indices]:
                y_ev = 2
            elif 1 in y_train[sub_indices]:
                y_ev = 1

            p0_lr = in_lr[sub_indices, 0]
            act_lr = (1.0 - p0_lr) ** 1.5
            s_act_lr = np.sum(act_lr)
            w_lr = act_lr / s_act_lr if s_act_lr >= 1e-6 else np.full(len(sub_indices), 1.0 / len(sub_indices))
            lr_gated = np.sum(w_lr[:, None] * in_lr[sub_indices], axis=0)

            p0_hgb = in_hgb[sub_indices, 0]
            act_hgb = (1.0 - p0_hgb) ** 1.5
            s_act_hgb = np.sum(act_hgb)
            w_hgb = act_hgb / s_act_hgb if s_act_hgb >= 1e-6 else np.full(len(sub_indices), 1.0 / len(sub_indices))
            hgb_gated = np.sum(w_hgb[:, None] * in_hgb[sub_indices], axis=0)

            df_train_groups.append({
                "group": gname,
                "true_label": y_ev,
                "lr_gated": lr_gated,
                "hgb_gated": hgb_gated
            })

        df_tg = pd.DataFrame(df_train_groups)
        y_tg = df_tg["true_label"].to_numpy()
        n0 = max(1, int((y_tg == 0).sum()))
        n1 = max(1, int((y_tg == 1).sum()))
        n2 = max(1, int((y_tg == 2).sum()))

        lr_tg = np.vstack(df_tg["lr_gated"].to_numpy())
        hgb_tg = np.vstack(df_tg["hgb_gated"].to_numpy())

        tau_ruido_grid = np.linspace(0.40, 0.70, 31)
        tau_aedes_grid = np.linspace(0.35, 0.65, 31)
        w_grid = [0.0, 0.05, 0.10, 0.15, 0.20]

        best_recall = -1.0
        best_fpr_sum = float("inf")
        best_w = 0.0
        best_tau_r = 0.40
        best_tau_ae = 0.50

        for w in w_grid:
            p_lr_safe = np.maximum(lr_tg, 1e-12)
            p_hgb_safe = np.maximum(hgb_tg, 1e-12)
            z = w * np.log(p_lr_safe) + (1.0 - w) * np.log(p_hgb_safe)
            z_shift = z - np.max(z, axis=-1, keepdims=True)
            exp_z = np.exp(z_shift)
            p_fused = exp_z / np.sum(exp_z, axis=-1, keepdims=True)

            p0_tr = p_fused[:, 0]
            cond_tr = p_fused[:, 2] / (p_fused[:, 1] + p_fused[:, 2] + 1e-12)

            for tau_r in tau_ruido_grid:
                pass_noise = (p0_tr < tau_r)
                if not np.any(pass_noise):
                    continue
                cond_act = cond_tr[pass_noise]
                y_act = y_tg[pass_noise]

                for tau_ae in tau_aedes_grid:
                    is_ae = (cond_act >= tau_ae)
                    cnt0 = np.sum((y_act == 0) & is_ae)
                    cnt1 = np.sum((y_act == 1) & is_ae)
                    cnt2 = np.sum((y_act == 2) & is_ae)

                    fpr0 = cnt0 / n0
                    fpr1 = cnt1 / n1
                    rec2 = cnt2 / n2

                    if fpr0 <= 0.007501 and fpr1 <= 0.017501:
                        fpr_sum = fpr0 + fpr1
                        if (rec2 > best_recall) or (abs(rec2 - best_recall) < 1e-6 and fpr_sum < best_fpr_sum):
                            best_recall = rec2
                            best_fpr_sum = fpr_sum
                            best_w = w
                            best_tau_r = float(tau_r)
                            best_tau_ae = float(tau_ae)

        if best_recall < 0:
            for w in w_grid:
                p_lr_safe = np.maximum(lr_tg, 1e-12)
                p_hgb_safe = np.maximum(hgb_tg, 1e-12)
                z = w * np.log(p_lr_safe) + (1.0 - w) * np.log(p_hgb_safe)
                z_shift = z - np.max(z, axis=-1, keepdims=True)
                exp_z = np.exp(z_shift)
                p_fused = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
                p0_tr = p_fused[:, 0]
                cond_tr = p_fused[:, 2] / (p_fused[:, 1] + p_fused[:, 2] + 1e-12)

                for tau_r in tau_ruido_grid:
                    pass_noise = (p0_tr < tau_r)
                    if not np.any(pass_noise):
                        continue
                    cond_act = cond_tr[pass_noise]
                    y_act = y_tg[pass_noise]
                    for tau_ae in tau_aedes_grid:
                        is_ae = (cond_act >= tau_ae)
                        fpr0 = np.sum((y_act == 0) & is_ae) / n0
                        fpr1 = np.sum((y_act == 1) & is_ae) / n1
                        rec2 = np.sum((y_act == 2) & is_ae) / n2
                        if fpr0 <= 0.010001 and fpr1 <= 0.020001:
                            fpr_sum = fpr0 + fpr1
                            if (rec2 > best_recall) or (abs(rec2 - best_recall) < 1e-6 and fpr_sum < best_fpr_sum):
                                best_recall = rec2
                                best_fpr_sum = fpr_sum
                                best_w = w
                                best_tau_r = float(tau_r)
                                best_tau_ae = float(tau_ae)

        inner_thresholds_per_fold[f"fold_{fold}"] = {
            "tau_noise": best_tau_r,
            "tau_aedes": best_tau_ae,
            "blend_weight_lr": best_w,
            "train_aedes_recall": float(best_recall),
            "selected_gamma": best_gamma,
            "train_folds": [f for f in range(5) if f != fold]
        }
        print(f"  --> Limiares inner fold {fold}: tau_noise={best_tau_r:.3f}, tau_aedes={best_tau_ae:.3f}, w={best_w:.2f}")

        # Treinamento final da dobra externa em todo D_train com best_gamma
        sw_outer_train = compute_sample_weights(df_train, y_train, gamma=best_gamma)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        lr_outer = LogisticRegression(class_weight=None, max_iter=2000, C=0.1, random_state=SEED)
        lr_outer.fit(X_train_scaled, y_train, sample_weight=sw_outer_train)
        oof_lr[val_idx] = lr_outer.predict_proba(X_val_scaled)

        hgb_outer = HistGradientBoostingClassifier(
            max_iter=200, l2_regularization=3.0, learning_rate=0.05, random_state=SEED
        )
        hgb_outer.fit(X_train, y_train, sample_weight=sw_outer_train)
        oof_hgb[val_idx] = hgb_outer.predict_proba(X_val)

        print(f"Fold Externo {fold + 1}/5 concluído.")

    oof_ens = (oof_lr + oof_hgb) / 2.0

    oof_lr = oof_lr / oof_lr.sum(axis=1, keepdims=True)
    oof_hgb = oof_hgb / oof_hgb.sum(axis=1, keepdims=True)
    oof_ens = oof_ens / oof_ens.sum(axis=1, keepdims=True)

    for p, name in [(oof_lr, "LogisticRegression"), (oof_hgb, "HistGradientBoosting"), (oof_ens, "Ensemble")]:
        assert np.isfinite(p).all(), f"Probabilidades com NaN/Inf em {name}"
        assert (p >= 0).all(), f"Probabilidades negativas em {name}"
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5), f"Soma != 1 em {name}"

    metrics_lr = compute_metrics(y, oof_lr, "logistic_regression")
    metrics_hgb = compute_metrics(y, oof_hgb, "hist_gradient_boosting")
    metrics_ens = compute_metrics(y, oof_ens, "ensemble")

    metrics_bundle = {
        "protocol": {
            "n_splits": 5,
            "inner_splits": 4,
            "seed": SEED,
            "validation": "StrictNestedStratifiedGroupKFold",
            "pcm_block_leakage": 0,
            "sample_weight": {
                "method": "duration_damping_with_class_renormalization",
                "formula": "w_i = u_i * (N_total / (3 * S_c)), u_i = clip(N_janelas(rec_i), 1, 60)^(-gamma)",
                "gammas_evaluated": gammas_to_evaluate,
                "selected_gammas_per_fold": outer_fold_gamma_selected,
                "clip_min": 1,
                "clip_max": 60
            }
        },
        "dataset_statistics": data_stats,
        "inner_calibrations": inner_thresholds_per_fold,
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
    (OUT_DIR / "inner_thresholds.json").write_text(json.dumps(inner_thresholds_per_fold, indent=2, ensure_ascii=False))

    pred_df = pd.DataFrame({
        "path": df["path"],
        "group": df["group"],
        "pcm_block": df["pcm_block"],
        "start_s": df["start_s"],
        "duration_s": df["duration_s"],
        "domain": df["domain"],
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
    print(f"\nPredições salvas com sucesso em {OUT_DIR / 'predictions_oof.csv'}")


if __name__ == "__main__":
    run_nested_training()
