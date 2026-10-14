"""
Avaliação em Nível de Evento (Agregação Temporal) da Detecção Ternária.
mosquito-wingbeat - 2026-10-14

Metodologia Estrita:
1. Auditoria de denominadores: 426 ruídos, 1014 outros, 64 aedes (1504 eventos totais).
2. Persistência temporal bounded por WAV individual (proibição estrita de cruzar arquivos distintos).
3. Activity-gated pooling: a_t = (1 - P_0(t))^1.5.
4. Logarithmic Opinion Pool (fusão log-pool geométrica ponderada normalizada).
5. Calibração Neyman-Pearson com aplicação cega aos eventos do fold externo de teste.
6. Registro de incerteza: Wilson score 95% CI e detalhamento por fonte (Dryad, HumBug, MosquitoSong).
"""

import sys
import json
import argparse
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
CLASS_NAMES = ["no_mosquito", "other_mosquito", "aedes_aegypti"]


def wilson_ci(k, n, confidence=0.95):
    """Intervalo de confiança de Wilson (95%) para proporções binomiais."""
    if n == 0:
        return {"point": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "ci_95": [0.0, 0.0]}
    z = 1.959963984540054
    p_hat = float(k) / float(n)
    denominator = 1.0 + (z ** 2) / n
    center_adj = p_hat + (z ** 2) / (2.0 * n)
    spread = z * np.sqrt((p_hat * (1.0 - p_hat) / n) + ((z ** 2) / (4.0 * (n ** 2))))
    lower = max(0.0, (center_adj - spread) / denominator)
    upper = min(1.0, (center_adj + spread) / denominator)
    return {
        "point": float(p_hat),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "ci_95": [float(lower), float(upper)]
    }


def softmax(z):
    z_arr = np.asarray(z)
    z_shift = z_arr - np.max(z_arr, axis=-1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def logarithmic_opinion_pool(p_lr, p_hgb, w=0.60, eps=1e-12):
    p_lr_safe = np.maximum(p_lr, eps)
    p_hgb_safe = np.maximum(p_hgb, eps)
    z = w * np.log(p_lr_safe) + (1.0 - w) * np.log(p_hgb_safe)
    p_fused = softmax(z)
    return p_fused / np.sum(p_fused, axis=-1, keepdims=True)


def audit_denominators(df):
    n_windows = {int(c): int((df["true_label"] == c).sum()) for c in range(3)}
    touching = {int(c): int(df[df["true_label"] == c]["group"].nunique()) for c in range(3)}

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

    return {
        "n_windows_per_class": n_windows,
        "n_recordings_touching_class": touching,
        "n_pure_events_per_class": pure_events
    }, event_labels_series


def activity_gated_pooling(probs):
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


def compute_group_wav_bounded_persistence(sub_df, p2_col="ensemble_prob_2"):
    """
    Persistência temporal delimitada estritamente por arquivo WAV individual:
    - Agrupa por path.
    - Ordena janelas por start_s.
    - Calcula média móvel de 2 janelas contíguas apenas se (start_s[i+1] - start_s[i]) <= 1.05.
    - Se o WAV tiver apenas 1 janela, persistência é o p2 dessa janela.
    - A persistência do grupo é o máximo entre as persistências de seus WAVs.
    - Proibição estrita: médias móveis jamais cruzam arquivos diferentes do mesmo group.
    """
    wav_persistences = []
    has_start = "start_s" in sub_df.columns
    for _, wav_df in sub_df.groupby("path"):
        if has_start:
            wav_df = wav_df.sort_values("start_s")
        p2 = wav_df[p2_col].to_numpy(dtype=np.float64)
        t = len(p2)
        if t == 1 or not has_start:
            wav_persistences.append(float(p2[0]))
        else:
            start_s = wav_df["start_s"].to_numpy(dtype=np.float64)
            valid_rolling = []
            for i in range(t - 1):
                if (start_s[i + 1] - start_s[i]) <= 1.05:
                    valid_rolling.append((p2[i] + p2[i + 1]) / 2.0)
            if len(valid_rolling) > 0:
                wav_persistences.append(float(np.max(valid_rolling)))
            else:
                wav_persistences.append(float(np.max(p2)))

    if len(wav_persistences) == 0:
        return 0.0
    return float(np.max(wav_persistences))


def get_canonical_source(domain_val, path_val=""):
    d = str(domain_val).lower()
    p = str(path_val).lower()
    if "dryad" in d or "dryad" in p:
        return "Dryad"
    elif "humbug" in d or "humbug" in p:
        return "HumBug"
    elif "mosquitosong" in d or "mosquitosong" in p:
        return "MosquitoSong"
    return "Outros"


def calibrate_neyman_pearson_oof(rec_df):
    """
    Calibração OOF para pipelines legados (retrocompatibilidade).
    """
    tau_ruido_grid = np.linspace(0.40, 0.70, 61)
    tau_aedes_grid = np.linspace(0.35, 0.65, 61)
    w_grid = [0.0, 0.05, 0.10, 0.15, 0.20]

    y_true = rec_df["true_label"].to_numpy(dtype=int)
    folds = rec_df["fold"].to_numpy(dtype=int)
    lr_gated_list = rec_df["lr_prob_gated"].to_numpy()
    hgb_gated_list = rec_df["hgb_prob_gated"].to_numpy()

    fused_by_w = {}
    for w in w_grid:
        fused = [
            logarithmic_opinion_pool(lr_gated_list[i], hgb_gated_list[i], w=w)
            for i in range(len(rec_df))
        ]
        fused_by_w[w] = np.array(fused)

    oof_preds = np.zeros(len(rec_df), dtype=int)
    thresholds_per_fold = {}

    for k in range(5):
        train_mask = (folds != k)
        val_mask = (folds == k)

        y_train = y_true[train_mask]
        n0 = max(1, int(np.sum(y_train == 0)))
        n1 = max(1, int(np.sum(y_train == 1)))
        n2 = max(1, int(np.sum(y_train == 2)))

        best_recall = -1.0
        best_w = 0.0
        best_tau_ruido = 0.40
        best_tau_aedes = 0.50
        best_fpr_sum = float("inf")

        for w in w_grid:
            p_train = fused_by_w[w][train_mask]
            p0_train = p_train[:, 0]
            cond_train = p_train[:, 2] / (p_train[:, 1] + p_train[:, 2] + 1e-12)

            for tau_r in tau_ruido_grid:
                pass_noise_mask = (p0_train < tau_r)
                if not np.any(pass_noise_mask):
                    continue

                cond_active = cond_train[pass_noise_mask]
                y_active = y_train[pass_noise_mask]

                for tau_ae in tau_aedes_grid:
                    is_ae = (cond_active >= tau_ae)
                    cnt0 = np.sum((y_active == 0) & is_ae)
                    cnt1 = np.sum((y_active == 1) & is_ae)
                    cnt2 = np.sum((y_active == 2) & is_ae)

                    fpr0 = cnt0 / n0
                    fpr1 = cnt1 / n1
                    rec2 = cnt2 / n2

                    if fpr0 <= 0.007501 and fpr1 <= 0.017501:
                        fpr_sum = fpr0 + fpr1
                        if (rec2 > best_recall) or (abs(rec2 - best_recall) < 1e-6 and fpr_sum < best_fpr_sum):
                            best_recall = rec2
                            best_fpr_sum = fpr_sum
                            best_w = w
                            best_tau_ruido = float(tau_r)
                            best_tau_aedes = float(tau_ae)

        if best_recall < 0:
            for w in w_grid:
                p_train = fused_by_w[w][train_mask]
                p0_train = p_train[:, 0]
                cond_train = p_train[:, 2] / (p_train[:, 1] + p_train[:, 2] + 1e-12)

                for tau_r in tau_ruido_grid:
                    pass_noise_mask = (p0_train < tau_r)
                    if not np.any(pass_noise_mask):
                        continue
                    cond_active = cond_train[pass_noise_mask]
                    y_active = y_train[pass_noise_mask]

                    for tau_ae in tau_aedes_grid:
                        is_ae = (cond_active >= tau_ae)
                        fpr0 = np.sum((y_active == 0) & is_ae) / n0
                        fpr1 = np.sum((y_active == 1) & is_ae) / n1
                        rec2 = np.sum((y_active == 2) & is_ae) / n2

                        if fpr0 <= 0.010001 and fpr1 <= 0.01973:
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

        p_val = fused_by_w[best_w][val_mask]
        cond_val = p_val[:, 2] / (p_val[:, 1] + p_val[:, 2] + 1e-12)
        val_is_noise = (p_val[:, 0] >= best_tau_ruido)
        val_is_ae = (~val_is_noise) & (cond_val >= best_tau_aedes)
        oof_preds[val_mask] = np.where(val_is_noise, 0, np.where(val_is_ae, 2, 1))

    return oof_preds, thresholds_per_fold


def evaluate_predictions(y_true, pred, prob=None, sources=None):
    acc = float(accuracy_score(y_true, pred))
    bacc = float(balanced_accuracy_score(y_true, pred))
    cm_raw = confusion_matrix(y_true, pred, labels=[0, 1, 2]).tolist()
    cm_norm = confusion_matrix(y_true, pred, labels=[0, 1, 2], normalize="true").tolist()

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, pred, labels=[0, 1, 2], zero_division=0
    )

    aedes_tp = int(cm_raw[2][2])
    aedes_total = int(sum(cm_raw[2]))
    noise_fp = int(cm_raw[0][2])
    noise_total = int(sum(cm_raw[0]))
    other_fp = int(cm_raw[1][2])
    other_total = int(sum(cm_raw[1]))

    out = {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "noise_to_aedes_fpr": float(cm_norm[0][2]),
        "other_to_aedes_fpr": float(cm_norm[1][2]),
        "aedes_recall": float(cm_norm[2][2]),
        "integer_counts": {
            "aedes_tp": aedes_tp,
            "aedes_total": aedes_total,
            "noise_fp_to_aedes": noise_fp,
            "noise_total": noise_total,
            "other_fp_to_aedes": other_fp,
            "other_total": other_total
        },
        "wilson_ci_95": {
            "aedes_recall": wilson_ci(aedes_tp, aedes_total),
            "noise_to_aedes_fpr": wilson_ci(noise_fp, noise_total),
            "other_to_aedes_fpr": wilson_ci(other_fp, other_total)
        },
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

    if sources is not None:
        source_breakdown = {}
        unique_sources = ["Dryad", "HumBug", "MosquitoSong"]
        for s in unique_sources:
            mask_s = (sources == s)
            y_s = y_true[mask_s]
            pred_s = pred[mask_s]

            s_ae_tot = int((y_s == 2).sum())
            s_ae_tp = int(((y_s == 2) & (pred_s == 2)).sum())

            s_noi_tot = int((y_s == 0).sum())
            s_noi_fp = int(((y_s == 0) & (pred_s == 2)).sum())

            s_oth_tot = int((y_s == 1).sum())
            s_oth_fp = int(((y_s == 1) & (pred_s == 2)).sum())

            source_breakdown[s] = {
                "aedes": {
                    "total": s_ae_tot,
                    "tp": s_ae_tp,
                    "recall": wilson_ci(s_ae_tp, s_ae_tot)
                },
                "other_mosquito": {
                    "total": s_oth_tot,
                    "fp": s_oth_fp,
                    "fpr": wilson_ci(s_oth_fp, s_oth_tot)
                },
                "no_mosquito": {
                    "total": s_noi_tot,
                    "fp": s_noi_fp,
                    "fpr": wilson_ci(s_noi_fp, s_noi_tot)
                }
            }
        out["source_breakdown"] = source_breakdown

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nested", action="store_true", help="Avalia resultados da pasta ternary_nested_2026_10_14")
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.results_dir:
        res_dir = args.results_dir
    elif args.nested:
        res_dir = ROOT / "results/ternary_nested_2026_10_14"
    else:
        res_dir = ROOT / "results/ternary_detection_2026_10_14"

    csv_path = res_dir / "predictions_oof.csv"
    out_json = res_dir / "event_level_metrics.json"

    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo {csv_path} não encontrado.")

    df = pd.read_csv(csv_path)
    audit_results, true_event_labels = audit_denominators(df)

    rec_records = []
    for group_name, sub in df.groupby("group", sort=False):
        fold = int(sub["fold"].iloc[0])
        y_ev = int(true_event_labels[group_name])
        t_len = len(sub)
        domain_val = sub["domain"].iloc[0] if "domain" in sub.columns else "dryad"
        source_name = get_canonical_source(domain_val, sub["path"].iloc[0])

        p_lr_mat = sub[["lr_prob_0", "lr_prob_1", "lr_prob_2"]].to_numpy()
        p_hgb_mat = sub[["hgb_prob_0", "hgb_prob_1", "hgb_prob_2"]].to_numpy()
        p_ens_mat = sub[["ensemble_prob_0", "ensemble_prob_1", "ensemble_prob_2"]].to_numpy()

        lr_gated = activity_gated_pooling(p_lr_mat)
        hgb_gated = activity_gated_pooling(p_hgb_mat)
        ens_gated = activity_gated_pooling(p_ens_mat)

        p_fused = logarithmic_opinion_pool(lr_gated, hgb_gated, w=0.60)
        rolling_max_p2 = compute_group_wav_bounded_persistence(sub, "ensemble_prob_2")

        rec_records.append({
            "group": group_name,
            "fold": fold,
            "true_label": y_ev,
            "true_class_name": CLASS_NAMES[y_ev],
            "source": source_name,
            "num_windows": t_len,
            "lr_prob_simple": p_lr_mat.mean(axis=0),
            "hgb_prob_simple": p_hgb_mat.mean(axis=0),
            "ens_prob_simple": p_ens_mat.mean(axis=0),
            "blend_prob_simple": 0.60 * p_lr_mat.mean(axis=0) + 0.40 * p_hgb_mat.mean(axis=0),
            "lr_prob_gated": lr_gated,
            "hgb_prob_gated": hgb_gated,
            "ens_prob_gated": ens_gated,
            "prob_fused": p_fused,
            "rolling_max_p2": rolling_max_p2
        })

    rec_df = pd.DataFrame(rec_records)
    y_true_events = rec_df["true_label"].to_numpy(dtype=int)
    sources = rec_df["source"].to_numpy(dtype=str)

    p_lr_all = np.vstack(rec_df["lr_prob_simple"].to_numpy())
    pred_lr = np.argmax(p_lr_all, axis=1)

    p_hgb_all = np.vstack(rec_df["hgb_prob_simple"].to_numpy())
    pred_hgb = np.argmax(p_hgb_all, axis=1)

    p_ens_all = np.vstack(rec_df["ens_prob_simple"].to_numpy())
    pred_ens = np.argmax(p_ens_all, axis=1)

    p_blend_all = np.vstack(rec_df["blend_prob_simple"].to_numpy())
    pred_blend = np.argmax(p_blend_all, axis=1)

    # Cascata Neyman-Pearson
    inner_thresh_file = res_dir / "inner_thresholds.json"
    pred_cascade = np.zeros(len(rec_df), dtype=int)
    fold_calibrations = {}

    if inner_thresh_file.exists():
        inner_thresh = json.loads(inner_thresh_file.read_text(encoding="utf-8"))
        for k in range(5):
            val_mask = (rec_df["fold"].to_numpy() == k)
            f_cfg = inner_thresh[f"fold_{k}"]
            tau_r = f_cfg["tau_noise"]
            tau_ae = f_cfg["tau_aedes"]
            w = f_cfg["blend_weight_lr"]

            lr_sub = np.vstack(rec_df.loc[val_mask, "lr_prob_gated"].to_numpy())
            hgb_sub = np.vstack(rec_df.loc[val_mask, "hgb_prob_gated"].to_numpy())
            p_val = logarithmic_opinion_pool(lr_sub, hgb_sub, w=w)

            cond_val = p_val[:, 2] / (p_val[:, 1] + p_val[:, 2] + 1e-12)
            val_is_noise = (p_val[:, 0] >= tau_r)
            val_is_ae = (~val_is_noise) & (cond_val >= tau_ae)
            pred_cascade[val_mask] = np.where(val_is_noise, 0, np.where(val_is_ae, 2, 1))

            fold_calibrations[f"fold_{k}"] = f_cfg
    else:
        pred_cascade, fold_calibrations = calibrate_neyman_pearson_oof(rec_df)

    models_data = {
        "logistic_regression_event": evaluate_predictions(y_true_events, pred_lr, p_lr_all, sources),
        "hist_gradient_boosting_event": evaluate_predictions(y_true_events, pred_hgb, p_hgb_all, sources),
        "ensemble_event": evaluate_predictions(y_true_events, pred_ens, p_ens_all, sources),
        "balanced_blend_event": evaluate_predictions(y_true_events, pred_blend, p_blend_all, sources),
        "activity_gated_log_odds_cascade_event": {
            **evaluate_predictions(y_true_events, pred_cascade, None, sources),
            "thresholds_per_fold": fold_calibrations
        }
    }

    output_payload = {
        "event_summary": {
            "total_recordings": len(rec_df),
            "recordings_per_class": {CLASS_NAMES[i]: int((y_true_events == i).sum()) for i in range(3)},
            "average_windows_per_recording": float(rec_df["num_windows"].mean()),
            "audit_denominators": {
                k: {str(cls_idx): val for cls_idx, val in v.items()}
                for k, v in audit_results.items()
            }
        },
        "models": models_data
    }

    out_json.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False))
    print(f"Relatório em nível de evento salvo com sucesso em {out_json}")

    print("\n=== RESUMO EM NÍVEL DE GRAVAÇÃO / EVENTO ===")
    for mname in [
        "logistic_regression_event",
        "hist_gradient_boosting_event",
        "ensemble_event",
        "activity_gated_log_odds_cascade_event"
    ]:
        m = models_data[mname]
        cnt = m["integer_counts"]
        w_rec = m["wilson_ci_95"]["aedes_recall"]
        w_fpr0 = m["wilson_ci_95"]["noise_to_aedes_fpr"]
        w_fpr1 = m["wilson_ci_95"]["other_to_aedes_fpr"]
        print(f"\n[{mname}]")
        print(f"  Acurácia Global: {m['accuracy']*100:.2f}% | Acurácia Balanceada: {m['balanced_accuracy']*100:.2f}%")
        print(f"  Aedes aegypti Recall: {m['aedes_recall']*100:.1f}% ({cnt['aedes_tp']}/{cnt['aedes_total']}) [95% CI: {w_rec['ci_lower']*100:.1f}% - {w_rec['ci_upper']*100:.1f}%]")
        print(f"  FPR Ruído -> Aedes: {m['noise_to_aedes_fpr']*100:.2f}% ({cnt['noise_fp_to_aedes']}/{cnt['noise_total']}) [95% CI: {w_fpr0['ci_lower']*100:.2f}% - {w_fpr0['ci_upper']*100:.2f}%]")
        print(f"  FPR Outro Mosquito -> Aedes: {m['other_to_aedes_fpr']*100:.2f}% ({cnt['other_fp_to_aedes']}/{cnt['other_total']}) [95% CI: {w_fpr1['ci_lower']*100:.2f}% - {w_fpr1['ci_upper']*100:.2f}%]")
        print(f"  Precisão Aedes: {m['per_class']['aedes_aegypti']['precision']*100:.2f}%")

        if "source_breakdown" in m:
            print("  --- Detalhamento por Fonte (Aedes TP/Total) ---")
            for sname, sdata in m["source_breakdown"].items():
                ae_info = sdata["aedes"]
                if ae_info["total"] > 0:
                    print(f"    {sname:12s}: {ae_info['tp']}/{ae_info['total']} ({ae_info['recall']['point']*100:.1f}%) [CI: {ae_info['recall']['ci_lower']*100:.1f}% - {ae_info['recall']['ci_upper']*100:.1f}%]")


if __name__ == "__main__":
    main()
