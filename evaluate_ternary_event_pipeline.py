"""
Avaliação em Nível de Evento (Agregação Temporal) da Detecção Ternária.
mosquito-wingbeat - 2026-10-14

Calcula métricas agregadas por gravação / evento de voo:
  - Média ponderada temporal
  - Comparativo Janela vs Evento
  - Calibração de limiares operacionais para vigilância sanitária
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results/ternary_detection_2026_10_14"
CSV_PATH = RESULTS_DIR / "predictions_oof.csv"
OUT_JSON = RESULTS_DIR / "event_level_metrics.json"

CLASS_NAMES = ["no_mosquito", "other_mosquito", "aedes_aegypti"]


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Arquivo {CSV_PATH} não encontrado.")

    df = pd.read_csv(CSV_PATH)
    groups = df["group"].unique()

    # Agregação por gravação (evento de áudio completo)
    rec_agg = df.groupby("group").agg({
        "true_label": "first",
        "true_class_name": "first",
        "lr_prob_0": "mean",
        "lr_prob_1": "mean",
        "lr_prob_2": "mean",
        "hgb_prob_0": "mean",
        "hgb_prob_1": "mean",
        "hgb_prob_2": "mean",
        "ensemble_prob_0": "mean",
        "ensemble_prob_1": "mean",
        "ensemble_prob_2": "mean",
        "path": "count"
    }).rename(columns={"path": "num_windows"}).reset_index()

    y_true = rec_agg["true_label"].to_numpy(int)

    # 1. Regressão Logística por Evento
    p_lr = rec_agg[["lr_prob_0", "lr_prob_1", "lr_prob_2"]].to_numpy()
    pred_lr = np.argmax(p_lr, axis=1)

    # 2. HistGradientBoosting por Evento
    p_hgb = rec_agg[["hgb_prob_0", "hgb_prob_1", "hgb_prob_2"]].to_numpy()
    pred_hgb = np.argmax(p_hgb, axis=1)

    # 3. Ensemble Soft Voting por Evento
    p_ens = rec_agg[["ensemble_prob_0", "ensemble_prob_1", "ensemble_prob_2"]].to_numpy()
    pred_ens = np.argmax(p_ens, axis=1)

    # 4. Blend Equilibrado de Alta Sensibilidade (60% LR + 40% HGB)
    p_blend = 0.60 * p_lr + 0.40 * p_hgb
    pred_blend = np.argmax(p_blend, axis=1)

    models_data = {}
    for name, pred, prob in [
        ("logistic_regression_event", pred_lr, p_lr),
        ("hist_gradient_boosting_event", pred_hgb, p_hgb),
        ("ensemble_event", pred_ens, p_ens),
        ("balanced_blend_event", pred_blend, p_blend)
    ]:
        acc = float(accuracy_score(y_true, pred))
        bacc = float(balanced_accuracy_score(y_true, pred))
        cm_raw = confusion_matrix(y_true, pred, labels=[0, 1, 2]).tolist()
        cm_norm = confusion_matrix(y_true, pred, labels=[0, 1, 2], normalize="true").tolist()

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, pred, labels=[0, 1, 2], zero_division=0
        )

        models_data[name] = {
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

    output_payload = {
        "event_summary": {
            "total_recordings": len(rec_agg),
            "recordings_per_class": {
                CLASS_NAMES[i]: int((y_true == i).sum()) for i in range(3)
            },
            "average_windows_per_recording": float(rec_agg["num_windows"].mean())
        },
        "models": models_data
    }

    OUT_JSON.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False))
    print(f"Relatório em nível de evento salvo com sucesso em {OUT_JSON}")

    # Exibição resumida
    print("\n=== RESUMO EM NÍVEL DE GRAVAÇÃO / EVENTO ===")
    for mname in ["logistic_regression_event", "hist_gradient_boosting_event", "ensemble_event", "balanced_blend_event"]:
        m = models_data[mname]
        print(f"\n[{mname}]")
        print(f"  Acurácia Global: {m['accuracy']*100:.2f}% | Acurácia Balanceada: {m['balanced_accuracy']*100:.2f}%")
        print(f"  Aedes aegypti Recall (Sensibilidade): {m['aedes_recall']*100:.1f}%")
        print(f"  Falso Alarme Ruído -> Aedes: {m['noise_to_aedes_fpr']*100:.2f}%")
        print(f"  Falso Alarme Outro Mosquito -> Aedes: {m['other_to_aedes_fpr']*100:.2f}%")


if __name__ == "__main__":
    main()
