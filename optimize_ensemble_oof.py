"""Otimização de Ensemble Out-of-Fold (OOF) Estritamente Livre de Vazamento.

Combina modelos de representações espectrais distintas (F0 acústico, MFCC,
ExtraTrees, SVM RBF e HistGradientBoosting com contraste) através de
otimização out-of-fold. Em cada dobra k de teste, os pesos da mistura e
a calibração de priors por classe são ajustados unicamente nas dobras
de treino (j != k), garantindo zero contaminação entre treino e teste.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/improvements"
OUT.mkdir(parents=True, exist_ok=True)

def main():
    segments = pd.read_csv(ROOT / "results/segments.csv")
    pos_mask = segments.label.ne("noise").to_numpy()
    pos_segments = segments[pos_mask].copy()
    classes = sorted(pos_segments.label.unique().tolist())
    y = pos_segments.label.map({c: i for i, c in enumerate(classes)}).to_numpy()
    folds = pos_segments.fold_species.to_numpy()
    groups = pos_segments.group.to_numpy()

    legacy = np.load(ROOT / "results/species_oof_probabilities.npz")
    improved = np.load(ROOT / "results/improvements/species_oof.npz")

    candidate_models = [
        ("F0_Likelihood", legacy["model_2"]),
        ("MFCC_LogReg", legacy["model_4"]),
        ("ExtraTrees_Base", legacy["model_5"]),
        ("SVM_RBF", improved["model_0"]),
        ("ExtraTrees_Contrast", improved["model_1"]),
        ("HistGB_Contrast", improved["model_2"]),
        ("MLP_Contrast", improved["model_3"]),
    ]
    K = len(candidate_models)
    num_classes = len(classes)
    oof_pred = np.zeros((len(y), num_classes), dtype="float32")
    fold_weights = {}

    for test_fold in range(3):
        train_idx = np.where(folds != test_fold)[0]
        test_idx = np.where(folds == test_fold)[0]
        
        y_train = y[train_idx]
        class_counts = np.bincount(y_train, minlength=num_classes)
        sample_weights = 1.0 / np.maximum(class_counts[y_train], 1)
        sample_weights /= np.mean(sample_weights)
        
        train_probs = [m[1][train_idx] for m in candidate_models]
        
        def loss(params):
            w = np.exp(params[:K]) / np.sum(np.exp(params[:K]))
            log_prior = params[K:]
            blend = sum(w[i] * train_probs[i] for i in range(K))
            logits = np.log(np.clip(blend, 1e-12, 1.0)) + 0.4 * log_prior
            z = logits - logits.max(axis=1, keepdims=True)
            p = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
            p_correct = np.clip(p[np.arange(len(y_train)), y_train], 1e-12, 1.0)
            unweighted_ll = -np.mean(np.log(p_correct))
            weighted_ll = -np.mean(sample_weights * np.log(p_correct))
            return 0.5 * unweighted_ll + 0.5 * weighted_ll

        init_params = np.zeros(K + num_classes)
        res = minimize(loss, init_params, method="Powell", options={"maxiter": 25})
        
        w_opt = (np.exp(res.x[:K]) / np.sum(np.exp(res.x[:K]))).tolist()
        log_prior_opt = res.x[K:]
        fold_weights[f"fold_{test_fold}"] = {name: float(w_opt[i]) for i, (name, _) in enumerate(candidate_models)}
        
        test_blend = sum((w_opt[i] * candidate_models[i][1][test_idx]) for i in range(K))
        test_logits = np.log(np.clip(test_blend, 1e-12, 1.0)) + 0.4 * log_prior_opt
        test_z = test_logits - test_logits.max(axis=1, keepdims=True)
        test_p = np.exp(test_z) / np.sum(np.exp(test_z), axis=1, keepdims=True)
        oof_pred[test_idx] = test_p

    pred_class = oof_pred.argmax(axis=1)
    acc = float(accuracy_score(y, pred_class))
    bal_acc = float(balanced_accuracy_score(y, pred_class))
    macro_f1 = float(f1_score(y, pred_class, average="macro"))

    source_p = pd.DataFrame(oof_pred, index=groups).groupby(level=0, sort=False).mean()
    source_y = pos_segments.groupby("group", sort=False).label.first().map({c: i for i, c in enumerate(classes)})
    source_y = source_y.reindex(source_p.index).to_numpy()
    rec_recall = float(balanced_accuracy_score(source_y, source_p.to_numpy().argmax(axis=1)))

    metrics = [
        {
            "model": "Ensemble OOF Calibrado (Joint Balanced/Macro)",
            "task": "species",
            "n_classes": 20,
            "n_windows": len(y),
            "n_groups": int(pos_segments.group.nunique()),
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1,
            "recording_macro_recall": rec_recall,
        }
    ]
    pd.DataFrame(metrics).to_csv(OUT / "calibrated_ensemble_metrics.csv", index=False)
    np.savez_compressed(OUT / "calibrated_ensemble_oof.npz", probabilities=oof_pred, classes=classes)
    (OUT / "calibrated_ensemble_weights.json").write_text(json.dumps(fold_weights, indent=2))

    print("Resultados Salvos com Sucesso:")
    print(f"  Acurácia Geral:           {acc * 100:.2f}%")
    print(f"  Acurácia Balanceada:      {bal_acc * 100:.2f}%")
    print(f"  Macro Recall por Grupo:   {rec_recall * 100:.2f}%")
    print(f"  Macro F1:                 {macro_f1 * 100:.2f}%")

if __name__ == "__main__":
    main()
