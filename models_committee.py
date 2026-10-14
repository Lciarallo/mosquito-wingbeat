"""Committee and Ensemble Classifiers for Mosquito Wingbeat Detection.

Strategies:
1. Calibrated Soft-Voting Committee:
   Combines LightGBM, HistGradientBoosting, ExtraTrees, and RandomForest.
2. Stacking Meta-Classifier:
   Trained on recording-disjoint OOF raw probabilities, with the same scale at inference.
   Archived bundles without input_mode preserve their historical calibrated inputs.
"""
from pathlib import Path
import json
import numpy as np, pandas as pd, joblib
import torch
from sklearn.linear_model import LogisticRegression
from scipy.special import expit

from aedes_binary import ROOT, decisions
from aedes_classical import raw_scores
from bioacoustic_invariance import BIOACOUSTIC_FEATURE_NAMES

COMMITTEE_DIR = ROOT / 'results/corrected_2026_10_01/models_committee'
COMMITTEE_DIR.mkdir(parents=True, exist_ok=True)

class StackingMetaCommittee:
    def __init__(self, base_models: dict, meta_clf: LogisticRegression, weights_vector=None, input_mode='raw'):
        self.base_models = base_models
        self.meta_clf = meta_clf
        self.weights_vector = weights_vector
        self.input_mode = input_mode

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Collect base-model predictions on the scale used during OOF training.
        preds = []
        for name, m_dict in self.base_models.items():
            model = m_dict['model']
            if getattr(self, 'input_mode', 'calibrated') == 'raw':
                # Match exactly the uncalibrated probabilities used in grouped OOF training.
                p = model.predict_proba(X)[:, 1]
            else:
                # Existing archived bundles retain their original inference semantics.
                z = raw_scores(model, X)
                p = expit(m_dict['slope'] * z + m_dict['bias'])
            preds.append(p)
        stack_x = np.column_stack(preds)
        meta_p = self.meta_clf.predict_proba(stack_x)[:, 1]
        return meta_p

class SoftVotingCommittee:
    def __init__(self, base_models: dict, weights: dict[str, float] = None):
        self.base_models = base_models
        if weights is None:
            # Default equal weighting
            n = len(base_models)
            self.weights = {k: 1.0 / n for k in base_models}
        else:
            total = sum(weights.values())
            self.weights = {k: v / total for k, v in weights.items()}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        blended = np.zeros(len(X), dtype=np.float32)
        for name, m_dict in self.base_models.items():
            w = self.weights[name]
            model = m_dict['model']
            slope, bias = m_dict['slope'], m_dict['bias']
            z = raw_scores(model, X)
            p = expit(slope * z + bias)
            blended += w * p
        return blended

def build_and_calibrate_committees():
    if (COMMITTEE_DIR.parent / 'freeze.json').exists():
        raise FileExistsError('Do not refit a committee after its external evaluation.')
    print('=== BUILDING AND CALIBRATING CLASSIFIER COMMITTEES ===', flush=True)
    from models_trees import TREE_MODELS_DIR
    tree_dir = TREE_MODELS_DIR
    oof_path = tree_dir / 'oof_train_predictions.csv'
    if not oof_path.exists():
        raise FileNotFoundError('OOF predictions not found. Run models_trees.py first.')

    oof_df = pd.read_csv(oof_path)
    base_names = ['lightgbm', 'hist_gradient_boosting', 'extra_trees', 'random_forest']
    assert oof_df.groupby('group').fold.nunique().max() == 1
    y_tr = oof_df['target'].to_numpy()

    # Load base tree models
    base_models = {}
    for name in base_names:
        base_models[name] = joblib.load(tree_dir / f'{name}.joblib')

    # 1. Fit Stacking Meta-Classifier
    print('Fitting Stacking Meta-Classifier on Out-Of-Fold predictions...', flush=True)
    oof_X = oof_df[base_names].to_numpy()
    meta_clf = LogisticRegression(C=1.0, penalty='l2', max_iter=1000, random_state=20261004)
    meta_clf.fit(oof_X, y_tr)
    coefs = dict(zip(base_names, [round(float(c), 4) for c in meta_clf.coef_[0]]))
    print('Stacking Meta-Learner coefficients:', coefs, flush=True)

    stacking_committee = StackingMetaCommittee(base_models, meta_clf)
    
    # 2. Build Soft-Voting Committee with balanced domain-robust weights
    # ExtraTrees and LightGBM get highest weight due to complementary inductive biases
    custom_weights = {
        'lightgbm': 0.35,
        'extra_trees': 0.35,
        'hist_gradient_boosting': 0.20,
        'random_forest': 0.10
    }
    soft_committee = SoftVotingCommittee(base_models, weights=custom_weights)

    # 3. Calibrate Committees on Calibration Split
    f = pd.read_csv(ROOT / 'results/multisource/segments.csv', low_memory=False)
    from aedes_binary import targets, weights as get_weights
    y = targets(f)
    ca_mask = f.split.eq('calibration') & f.quality_ok & (y >= 0)
    ca_idx = np.flatnonzero(ca_mask)
    cw = get_weights(f.iloc[ca_idx])

    base_feats = np.load(ROOT / 'data/cache/aedes_classical_features.npy')
    bio_feats = np.load(ROOT / 'data/cache/bioacoustic_invariance_features.npy')
    X_full = np.concatenate([base_feats, bio_feats], axis=1).astype(np.float32)
    X_ca = X_full[ca_idx]
    y_ca = y[ca_idx]

    # Stacking calibration
    stack_p_ca = stacking_committee.predict_proba(X_ca)
    stack_high = next(float(t) for t in np.r_[np.linspace(0.5, 0.999, 500), 1.0001]
                      if np.average(stack_p_ca[y_ca == 0] >= t, weights=cw[y_ca == 0]) <= 0.05)
    stack_low = 0.0
    for t in np.linspace(0.001, min(0.499, stack_high - 1e-6), 499):
        if np.average(stack_p_ca[y_ca == 1] <= t, weights=cw[y_ca == 1]) <= 0.05:
            stack_low = float(t)

    # Soft Voting calibration
    soft_p_ca = soft_committee.predict_proba(X_ca)
    soft_high = next(float(t) for t in np.r_[np.linspace(0.5, 0.999, 500), 1.0001]
                     if np.average(soft_p_ca[y_ca == 0] >= t, weights=cw[y_ca == 0]) <= 0.05)
    soft_low = 0.0
    for t in np.linspace(0.001, min(0.499, soft_high - 1e-6), 499):
        if np.average(soft_p_ca[y_ca == 1] <= t, weights=cw[y_ca == 1]) <= 0.05:
            soft_low = float(t)

    payload = {
        'stacking_committee': stacking_committee,
        'stacking_thresholds': {'low': stack_low, 'high': stack_high},
        'soft_voting_committee': soft_committee,
        'soft_voting_thresholds': {'low': soft_low, 'high': soft_high},
        'meta_coefficients': coefs
    }
    joblib.dump(payload, COMMITTEE_DIR / 'committees_bundle.joblib')
    print(f'Committees calibrated and saved. Stacking: [{stack_low:.4f}, {stack_high:.4f}], Soft: [{soft_low:.4f}, {soft_high:.4f}]', flush=True)
    return payload

if __name__ == '__main__':
    build_and_calibrate_committees()
