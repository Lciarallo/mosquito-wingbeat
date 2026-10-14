"""
Avaliação em Nível de Evento (Agregação Temporal) da Detecção Ternária.
mosquito-wingbeat - 2026-10-14

Calcula métricas agregadas por gravação / evento de voo:
  - Auditoria de Denominadores (puros vs tocados)
  - Activity-Gated Pooling (combate à diluição por janelas de silêncio)
  - Logarithmic Opinion Pool (fusão log-pool geométrica ponderada normalizada)
  - Calibração de Cascata Neyman-Pearson OOF Estrita (Zero Leakage)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support
)

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results/ternary_detection_2026_10_14"
CSV_PATH = RESULTS_DIR / "predictions_oof.csv"
OUT_JSON = RESULTS_DIR / "event_level_metrics.json"

CLASS_NAMES = ["no_mosquito", "other_mosquito", "aedes_aegypti"]


def softmax(z):
    """Calcula softmax numérico com garantia estrita de soma 1.0."""
    z_arr = np.asarray(z)
    z_shift = z_arr - np.max(z_arr, axis=-1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def logarithmic_opinion_pool(p_lr, p_hgb, w=0.60, eps=1e-12):
    """
    Logarithmic Opinion Pool (fusão logarítmica com garantia estrita de soma 1.0):
      z_c = w * log(max(P_c_LR, eps)) + (1 - w) * log(max(P_c_HGB, eps))
      P_c = softmax(z)
    Equivale à média geométrica ponderada normalizada: P_c ~ (P_c_LR)^w * (P_c_HGB)^(1-w).
    """
    p_lr_safe = np.maximum(p_lr, eps)
    p_hgb_safe = np.maximum(p_hgb, eps)

    z = w * np.log(p_lr_safe) + (1.0 - w) * np.log(p_hgb_safe)
    p_fused = softmax(z)
    # Normalização final para garantia exata de soma 1.0
    p_fused = p_fused / np.sum(p_fused, axis=-1, keepdims=True)
    return p_fused


def audit_denominators(df):
    """
    Realiza a auditoria estrita de denominadores:
      - n_windows_per_class: {0: 8658, 1: 33332, 2: 2778}
      - n_recordings_touching_class: {0: 471, 1: 1014, 2: 64}
      - n_pure_events_per_class: {0: 426, 1: 1014, 2: 64} (soma 1504)
    Rótulo hierárquico verdadeiro: se tem janela 2 -> 2; senão se tem 1 -> 1; senão -> 0.
    """
    n_windows = {int(c): int((df["true_label"] == c).sum()) for c in range(3)}

    touching = {
        int(c): int(df[df["true_label"] == c]["group"].nunique())
        for c in range(3)
    }

    # Rotulagem epidemiológica hierárquica por evento
    group_labels = df.groupby("group")["true_label"].apply(set)

    def assign_event(labels_set):
        if 2 in labels_set:
            return 2
        elif 1 in labels_set:
            return 1
        return 0

    event_labels_series = group_labels.apply(assign_event)
    pure_events = {int(c): int((event_labels_series == c).sum()) for c in range(3)}

    assert sum(pure_events.values()) == 1504, f"Soma dos eventos puros difere de 1504: {sum(pure_events.values())}"
    assert pure_events == {0: 426, 1: 1014, 2: 64}, f"Divergência nos eventos puros: {pure_events}"
    assert touching == {0: 471, 1: 1014, 2: 64}, f"Divergência nas gravações que tocam classes: {touching}"
    assert n_windows == {0: 8658, 1: 33332, 2: 2778}, f"Divergência nas janelas: {n_windows}"

    return {
        "n_windows_per_class": n_windows,
        "n_recordings_touching_class": touching,
        "n_pure_events_per_class": pure_events
    }, event_labels_series


def activity_gated_pooling(probs):
    """
    Activity-Gated Pooling:
      a_t = (1 - P_0(t))^1.5
      Se sum(a_t) < 1e-6: pesos uniformes 1/T. Senão: w_t = a_t / sum(a_t).
      Retorna prob média ponderada normalizada para soma 1.0.
    """
    p0 = probs[:, 0]
    act = (1.0 - p0) ** 1.5
    s_act = np.sum(act)
    t = len(probs)
    if s_act < 1e-6:
        w = np.full(t, 1.0 / t, dtype=np.float64)
    else:
        w = act / s_act
    pooled = np.sum(w[:, None] * probs, axis=0)
    return pooled / np.sum(pooled)


def compute_recording_descriptors(p_lr, p_hgb, p_ens):
    """
    Calcula descritores de gravação:
      - Pooling por atividade para LR e HistGB
      - Top-k de P2 e de atividade (1 - P0) com k = min(5, max(1, ceil(0.25 * T)))
      - Quantil P75 de P2
    """
    t = len(p_lr)
    k = min(5, max(1, int(np.ceil(0.25 * t))))

    lr_pooled = activity_gated_pooling(p_lr)
    hgb_pooled = activity_gated_pooling(p_hgb)

    p2_ens = p_ens[:, 2]
    topk_p2 = float(np.sort(p2_ens)[-k:].mean())
    topk_act = float(np.sort(1.0 - p_ens[:, 0])[-k:].mean())
    p75_p2 = float(np.percentile(p2_ens, 75))

    return lr_pooled, hgb_pooled, {
        "topk_p2": topk_p2,
        "topk_activity": topk_act,
        "p75_p2": p75_p2
    }


def calibrate_neyman_pearson_oof(rec_df):
    """
    Calibra limiares da cascata (tau_ruido, tau_aedes) e peso do blend estritamente OOF por dobra:
      Gate 1 (Rejeição de Ruído): Se P(no_mosquito) >= tau_ruido -> 0
      Gate 2 (Detecção de Vetor): Caso contrário, se P(aedes) / (P(aedes) + P(other) + eps) >= tau_aedes -> 2; senão -> 1
      Restrições: FPR_ruido <= 1.0% e FPR_outro <= 2.0%
      Objetivo: Maximizar Recall de Aedes
    Zero vazamento: fold k é avaliado usando estritamente os limiares dos outros 4 folds.
    """
    tau_ruido_grid = np.linspace(0.40, 0.80, 21)
    tau_aedes_grid = np.linspace(0.20, 0.60, 41)
    w_grid = [0.0, 0.05, 0.10, 0.15, 0.20]

    y_true = rec_df["true_label"].to_numpy(dtype=int)
    folds = rec_df["fold"].to_numpy(dtype=int)
    lr_gated_list = rec_df["lr_prob_gated"].to_numpy()
    hgb_gated_list = rec_df["hgb_prob_gated"].to_numpy()

    # Pre-calcula fused para cada w no grid
    fused_by_w = {}
    for w in w_grid:
        fused = [logarithmic_opinion_pool(lr_gated_list[i], hgb_gated_list[i], w=w) for i in range(len(rec_df))]
        fused_by_w[w] = np.array(fused)

    oof_preds = np.zeros(len(rec_df), dtype=int)
    thresholds_per_fold = {}

    for k in range(5):
        train_mask = (folds != k)
        val_mask = (folds == k)

        y_train = y_true[train_mask]
        n0 = max(1, np.sum(y_train == 0))
        n1 = max(1, np.sum(y_train == 1))
        n2 = max(1, np.sum(y_train == 2))

        best_recall = -1.0
        best_w = 0.0
        best_tau_ruido = 0.40
        best_tau_aedes = 0.46
        best_fpr_sum = float("inf")

        for w in w_grid:
            p_train = fused_by_w[w][train_mask]
            cond_train = p_train[:, 2] / (p_train[:, 1] + p_train[:, 2] + 1e-12)

            for tau_r in tau_ruido_grid:
                is_noise = (p_train[:, 0] >= tau_r)
                for tau_ae in tau_aedes_grid:
                    is_ae = (~is_noise) & (cond_train >= tau_ae)
                    pred_c = np.where(is_noise, 0, np.where(is_ae, 2, 1))

                    fpr0 = np.sum((y_train == 0) & (pred_c == 2)) / n0
                    fpr1 = np.sum((y_train == 1) & (pred_c == 2)) / n1
                    rec2 = np.sum((y_train == 2) & (pred_c == 2)) / n2

                    if fpr0 <= 0.010001 and fpr1 <= 0.020001:
                        fpr_sum = fpr0 + fpr1
                        if (rec2 > best_recall) or (abs(rec2 - best_recall) < 1e-6 and fpr_sum < best_fpr_sum):
                            best_recall = rec2
                            best_fpr_sum = fpr_sum
                            best_w = w
                            best_tau_ruido = float(tau_r)
                            best_tau_aedes = float(tau_ae)

        thresholds_per_fold[f"fold_{k}"] = {
            "tau_noise": best_tau_ruido,
            "tau_aedes": best_tau_aedes,
            "blend_weight_lr": best_w,
            "train_aedes_recall": float(best_recall),
            "train_folds": [f for f in range(5) if f != k]
        }

        # Aplicação cega aos eventos da dobra de validação
        p_val = fused_by_w[best_w][val_mask]
        cond_val = p_val[:, 2] / (p_val[:, 1] + p_val[:, 2] + 1e-12)
        val_is_noise = (p_val[:, 0] >= best_tau_ruido)
        val_is_ae = (~val_is_noise) & (cond_val >= best_tau_aedes)
        oof_preds[val_mask] = np.where(val_is_noise, 0, np.where(val_is_ae, 2, 1))

    return oof_preds, thresholds_per_fold


def evaluate_predictions(y_true, pred, prob=None):
    acc = float(accuracy_score(y_true, pred))
    bacc = float(balanced_accuracy_score(y_true, pred))
    cm_raw = confusion_matrix(y_true, pred, labels=[0, 1, 2]).tolist()
    cm_norm = confusion_matrix(y_true, pred, labels=[0, 1, 2], normalize="true").tolist()

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, pred, labels=[0, 1, 2], zero_division=0
    )

    return {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "noise_to_aedes_fpr": float(cm_norm[0][2]),
        "other_to_aedes_fpr": float(cm_norm[1][2]),
        "aedes_recall": float(cm_norm[2][2]),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i])
            } for i in range(3)
        },
        "confusion_matrix_raw": cm_raw,
        "confusion_matrix_norm": cm_norm
    }


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Arquivo {CSV_PATH} não encontrado.")

    df = pd.read_csv(CSV_PATH)

    # 1. Auditoria de Denominadores
    audit_results, true_event_labels = audit_denominators(df)

    rec_records = []
    for group_name, sub in df.groupby("group", sort=False):
        fold = int(sub["fold"].iloc[0])
        y_ev = int(true_event_labels[group_name])
        t_len = len(sub)

        p_lr_mat = sub[["lr_prob_0", "lr_prob_1", "lr_prob_2"]].to_numpy()
        p_hgb_mat = sub[["hgb_prob_0", "hgb_prob_1", "hgb_prob_2"]].to_numpy()
        p_ens_mat = sub[["ensemble_prob_0", "ensemble_prob_1", "ensemble_prob_2"]].to_numpy()

        lr_gated, hgb_gated, desc = compute_recording_descriptors(p_lr_mat, p_hgb_mat, p_ens_mat)

        # Fusão Logarithmic Opinion Pool (w=0.60 LR + 0.40 HGB)
        p_fused = logarithmic_opinion_pool(lr_gated, hgb_gated, w=0.60)

        # Médias simples para baselines existentes
        p_lr_simple = p_lr_mat.mean(axis=0)
        p_hgb_simple = p_hgb_mat.mean(axis=0)
        p_ens_simple = p_ens_mat.mean(axis=0)
        p_blend_simple = 0.60 * p_lr_simple + 0.40 * p_hgb_simple

        rec_records.append({
            "group": group_name,
            "fold": fold,
            "true_label": y_ev,
            "true_class_name": CLASS_NAMES[y_ev],
            "num_windows": t_len,
            "lr_prob_simple": p_lr_simple,
            "hgb_prob_simple": p_hgb_simple,
            "ens_prob_simple": p_ens_simple,
            "blend_prob_simple": p_blend_simple,
            "lr_prob_gated": lr_gated,
            "hgb_prob_gated": hgb_gated,
            "prob_fused": p_fused,
            **desc
        })

    rec_df = pd.DataFrame(rec_records)
    y_true_events = rec_df["true_label"].to_numpy(dtype=int)

    # Baselines existentes por evento
    p_lr_all = np.vstack(rec_df["lr_prob_simple"].to_numpy())
    pred_lr = np.argmax(p_lr_all, axis=1)

    p_hgb_all = np.vstack(rec_df["hgb_prob_simple"].to_numpy())
    pred_hgb = np.argmax(p_hgb_all, axis=1)

    p_ens_all = np.vstack(rec_df["ens_prob_simple"].to_numpy())
    pred_ens = np.argmax(p_ens_all, axis=1)

    p_blend_all = np.vstack(rec_df["blend_prob_simple"].to_numpy())
    pred_blend = np.argmax(p_blend_all, axis=1)

    # 4. Activity-Gated Log-Pool Neyman-Pearson Cascade Event (Nova Abordagem)
    pred_cascade, fold_calibrations = calibrate_neyman_pearson_oof(rec_df)

    models_data = {
        "logistic_regression_event": evaluate_predictions(y_true_events, pred_lr, p_lr_all),
        "hist_gradient_boosting_event": evaluate_predictions(y_true_events, pred_hgb, p_hgb_all),
        "ensemble_event": evaluate_predictions(y_true_events, pred_ens, p_ens_all),
        "balanced_blend_event": evaluate_predictions(y_true_events, pred_blend, p_blend_all),
        "activity_gated_log_odds_cascade_event": {
            **evaluate_predictions(y_true_events, pred_cascade),
            "thresholds_per_fold": fold_calibrations
        }
    }

    output_payload = {
        "event_summary": {
            "total_recordings": len(rec_df),
            "recordings_per_class": {
                CLASS_NAMES[i]: int((y_true_events == i).sum()) for i in range(3)
            },
            "average_windows_per_recording": float(rec_df["num_windows"].mean()),
            "audit_denominators": {
                k: {str(cls_idx): val for cls_idx, val in v.items()}
                for k, v in audit_results.items()
            }
        },
        "models": models_data
    }

    OUT_JSON.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False))
    print(f"Relatório em nível de evento salvo com sucesso em {OUT_JSON}")

    # Exibição resumida
    print("\n=== RESUMO EM NÍVEL DE GRAVAÇÃO / EVENTO ===")
    for mname in [
        "logistic_regression_event",
        "hist_gradient_boosting_event",
        "ensemble_event",
        "activity_gated_log_odds_cascade_event"
    ]:
        m = models_data[mname]
        print(f"\n[{mname}]")
        print(f"  Acurácia Global: {m['accuracy']*100:.2f}% | Acurácia Balanceada: {m['balanced_accuracy']*100:.2f}%")
        print(f"  Aedes aegypti Recall (Sensibilidade): {m['aedes_recall']*100:.1f}%")
        print(f"  Falso Alarme Ruído -> Aedes: {m['noise_to_aedes_fpr']*100:.2f}%")
        print(f"  Falso Alarme Outro Mosquito -> Aedes: {m['other_to_aedes_fpr']*100:.2f}%")


if __name__ == "__main__":
    main()
