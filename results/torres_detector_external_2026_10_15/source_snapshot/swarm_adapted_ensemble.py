"""
Fixed-reference ensemble for Aedes aegypti evaluation.

Combines:
1. DeepBioacousticHybridNet (waveform spectrogram + bioacoustic features)
2. Classical SVM RBF (smooth kernel boundaries in bioacoustic space)
3. Carrier energy ratio feature

Marginal score distributions are frozen on training data; thresholds are chosen
on a separate calibration partition. External labels are evaluation-only.
This fixes batch dependence; it does not establish invariance or reliability.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve
)

from aedes_binary import ROOT
from aedes_classical import raw_scores, features as extract_base_features
from bioacoustic_invariance import extract_batch_bioacoustic_features, BIOACOUSTIC_FEATURE_NAMES

class SwarmAdaptedEnsemble:
    def __init__(self, w_deep=0.50, w_svm=0.35, w_carrier=0.15):
        self.w_deep = w_deep
        self.w_svm = w_svm
        self.w_carrier = w_carrier
        self.calibrated_threshold = 0.50
        self.is_fitted = False
        self.reference_cdfs = None

    def fit_reference(self, components, sample_weight=None, *, partition):
        """Freeze marginal distributions using training recordings only."""
        if partition != 'train':
            raise ValueError('Reference distributions must be fitted on train only.')
        values = np.asarray(components, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
            raise ValueError('Expected finite (N, 3) component scores.')
        weight = np.ones(len(values)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if weight.shape != (len(values),) or (weight < 0).any() or not np.isfinite(weight).all() or weight.sum() <= 0:
            raise ValueError('Invalid training weights.')
        self.reference_cdfs = []
        for col in values.T:
            unique, inverse = np.unique(col, return_inverse=True)
            mass = np.bincount(inverse, weights=weight)
            midpoint = (np.cumsum(mass) - .5 * mass) / mass.sum()
            self.reference_cdfs.append((unique, midpoint))
        self.is_fitted = False
        return self

    def blend_components(self, components):
        """The same recording has the same score regardless of batch composition."""
        if getattr(self, 'reference_cdfs', None) is None:
            raise ValueError('No frozen training reference; legacy batch ranks are not supported.')
        values = np.asarray(components, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
            raise ValueError('Expected finite (N, 3) component scores.')
        transformed = np.column_stack([
            np.interp(values[:, i], x, q, left=0., right=1.)
            for i, (x, q) in enumerate(self.reference_cdfs)
        ])
        return transformed @ np.array([self.w_deep, self.w_svm, self.w_carrier])

    def predict_components(self, waves, base_feats, bio_feats, comb_feats):
        # 1. Deep model predictions
        import torch
        from models_deep import DeepBioacousticHybridNet
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        deep_path = ROOT / 'results/models_deep/deep_hybrid_net.pt'
        saved_deep = torch.load(deep_path, map_location=device, weights_only=False)
        deep_model = DeepBioacousticHybridNet(bio_features_dim=comb_feats.shape[1]).to(device)
        deep_model.load_state_dict(saved_deep['state_dict'])
        deep_model.eval()

        mean_sc = saved_deep['scaler_mean'].astype(np.float32)
        scale_sc = saved_deep['scaler_scale'].astype(np.float32)
        b_std = ((comb_feats - mean_sc) / scale_sc).astype(np.float32)

        # Pad/tile waves to SAMPLES (15872) if needed
        if waves.shape[1] < 15872:
            reps = int(np.ceil(15872 / waves.shape[1]))
            w_in = np.tile(waves, (1, reps))[:, :15872].astype(np.float32)
        else:
            w_in = waves[:, :15872].astype(np.float32)

        predictions = []
        with torch.no_grad():
            for start in range(0, len(w_in), 96):
                w_t = torch.from_numpy(w_in[start:start + 96]).to(device)
                b_t = torch.from_numpy(b_std[start:start + 96]).to(device)
                predictions.extend(deep_model(w_t, b_t, augment=False).cpu().numpy())
        deep_logits = np.asarray(predictions)
        p_deep = expit(saved_deep['slope'] * deep_logits + saved_deep['bias'])

        # 2. SVM RBF
        svm_dict = joblib.load(ROOT / 'results/aedes_classical/models/svm_rbf.joblib')
        z_svm = raw_scores(svm_dict['model'], base_feats)
        p_svm = expit(svm_dict['slope'] * z_svm + svm_dict['bias'])

        # 3. Carrier energy ratio feature
        c_idx = BIOACOUSTIC_FEATURE_NAMES.index('carrier_energy_ratio')
        carrier = bio_feats[:, c_idx]

        return p_deep, p_svm, carrier

    def predict_proba(self, waves, base_feats, bio_feats, comb_feats):
        p_deep, p_svm, carrier = self.predict_components(waves, base_feats, bio_feats, comb_feats)

        return self.blend_components(np.column_stack([p_deep, p_svm, carrier]))

    def calibrate_threshold(self, y_true, p_scores, *, partition, sample_weight=None):
        if partition != 'calibration':
            raise ValueError('Threshold must be chosen on a separate calibration partition.')
        fpr, tpr, thresh = roc_curve(y_true, p_scores, sample_weight=sample_weight)
        best_idx = np.argmax(tpr - fpr)
        self.calibrated_threshold = float(thresh[best_idx])
        self.is_fitted = True
        return self.calibrated_threshold

    def predict(self, waves, base_feats, bio_feats, comb_feats, threshold=None):
        p = self.predict_proba(waves, base_feats, bio_feats, comb_feats)
        t = threshold if threshold is not None else self.calibrated_threshold
        return (p >= t).astype(int), p

def main():
    # Fit on training/calibration partitions before evaluating frozen models.
    from corrected_validation import main as corrected_main
    return corrected_main()

if __name__ == '__main__':
    main()
