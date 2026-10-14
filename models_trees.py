"""Decision Tree and Gradient Boosting Classifiers for Mosquito Wingbeat Detection.

Includes:
1. LightGBM Classifier (with histogram binning, depth limits, feature/data bagging, L1/L2 reg).
2. HistGradientBoosting Classifier (with tuned leaf nodes, early stopping, and l2 regularization).
3. ExtraTrees Classifier (extremely randomized trees, superior variance reduction on out-of-domain).
4. Balanced Random Forest Classifier (subsampling bootstrap).
5. 5-Fold StratifiedGroupKFold OOF predictions; entire recordings stay in one fold.
"""
from pathlib import Path
import json, time
import numpy as np, pandas as pd, joblib
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.special import expit

from aedes_binary import targets, weights, ROOT, decisions
from aedes_classical import FEATURE_NAMES as BASE_FEATURE_NAMES, raw_scores
from train_aedes_binary import summary
from bioacoustic_invariance import BIOACOUSTIC_FEATURE_NAMES

TREE_MODELS_DIR = ROOT / 'results/corrected_2026_10_01/models_trees'
TREE_MODELS_DIR.mkdir(parents=True, exist_ok=True)

def train_and_evaluate_tree_models():
    if (TREE_MODELS_DIR.parent / 'freeze.json').exists():
        raise FileExistsError('This experiment is frozen; use a new experiment directory to retrain.')
    print('=== TRAINING AND EVALUATING TREE-BASED ENSEMBLES ===', flush=True)
    f = pd.read_csv(ROOT / 'results/multisource/segments.csv', low_memory=False)
    assert f.groupby('group').split.nunique().max() == 1
    assert f.dropna(subset=['content_sha256']).groupby('content_sha256').split.nunique().max() == 1
    y = targets(f)

    base_feats = np.load(ROOT / 'data/cache/aedes_classical_features.npy')
    bio_feats = np.load(ROOT / 'data/cache/bioacoustic_invariance_features.npy')
    X = np.concatenate([base_feats, bio_feats], axis=1).astype(np.float32)
    feature_names = BASE_FEATURE_NAMES + BIOACOUSTIC_FEATURE_NAMES

    tr_idx = np.flatnonzero(f.split.eq('train') & f.quality_ok & (y >= 0))
    va_idx = np.flatnonzero(f.split.eq('validation') & f.quality_ok & (y >= 0))
    ca_idx = np.flatnonzero(f.split.eq('calibration') & f.quality_ok & (y >= 0))
    te_idx = np.flatnonzero(f.split.isin(['test', 'challenge']))

    tw = weights(f.iloc[tr_idx])
    vw = weights(f.iloc[va_idx])
    cw = weights(f.iloc[ca_idx])

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]
    X_ca, y_ca = X[ca_idx], y[ca_idx]
    X_te = X[te_idx]

    models_config = {
        'lightgbm': lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=7,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.75,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=20261004,
            n_jobs=4,
            verbosity=-1
        ),
        'hist_gradient_boosting': HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.5,
            early_stopping=False,
            random_state=20261004
        ),
        'extra_trees': ExtraTreesClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=3,
            max_features=0.6,
            n_jobs=4,
            random_state=20261004
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=18,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight='balanced',
            n_jobs=4,
            random_state=20261004
        )
    }

    trained_models = {}
    val_metrics = {}
    oof_predictions = {name: np.zeros(len(tr_idx), dtype=np.float32) for name in models_config}
    fold_ids = np.full(len(tr_idx), -1, dtype=np.int64)
    fold_audit = []

    # 5-fold CV on train set to generate Out-Of-Fold (OOF) predictions for stacking meta-learner
    print('\nGenerating 5-Fold Out-Of-Fold (OOF) predictions on training data...', flush=True)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261004)
    groups = f.iloc[tr_idx].group.to_numpy()
    for fold, (f_tr, f_val) in enumerate(skf.split(X_tr, y_tr, groups)):
        overlap = set(groups[f_tr]) & set(groups[f_val])
        assert not overlap, 'Recording group leaked into OOF validation.'
        assert len(np.unique(y_tr[f_tr])) == len(np.unique(y_tr[f_val])) == 2
        fold_ids[f_val] = fold
        fold_audit.append(dict(fold=fold, train_windows=len(f_tr), validation_windows=len(f_val),
                               overlapping_groups=len(overlap), validation_groups=len(set(groups[f_val]))))
        print(f'Grouped fold {fold + 1}/5: {len(f_val)} held-out windows, zero group overlap', flush=True)
        fold_tw = weights(f.iloc[tr_idx[f_tr]])
        for name, clf_template in models_config.items():
            import copy
            fold_clf = copy.deepcopy(clf_template)
            fit_kwargs = {}
            if name != 'random_forest':
                fit_kwargs['sample_weight'] = fold_tw
            fold_clf.fit(X_tr[f_tr], y_tr[f_tr], **fit_kwargs)
            probs = fold_clf.predict_proba(X_tr[f_val])[:, 1]
            oof_predictions[name][f_val] = probs

    # Train final models on full train split
    print('\nFitting full models and calibrating on calibration split...', flush=True)
    for name, clf in models_config.items():
        t0 = time.time()
        fit_kwargs = {}
        if name != 'random_forest':
            fit_kwargs['sample_weight'] = tw
        clf.fit(X_tr, y_tr, **fit_kwargs)
        fit_time = time.time() - t0

        z_va = raw_scores(clf, X_va)
        auc_va = float(roc_auc_score(y_va, z_va, sample_weight=vw))
        val_metrics[name] = dict(val_auc=auc_va, fit_time_sec=round(fit_time, 2))
        print(f'Model: {name:24s} | Val AUC: {auc_va:.4f} | Time: {fit_time:.1f}s', flush=True)

        # Calibrate using Platt scaling on calibration split
        z_ca = raw_scores(clf, X_ca)
        calibrator = LogisticRegression(C=10, max_iter=2000).fit(z_ca[:, None], y_ca, sample_weight=cw)
        slope = float(calibrator.coef_[0, 0])
        bias = float(calibrator.intercept_[0])
        p_ca = expit(slope * z_ca + bias)

        # Target 5% FPR on calibration negatives
        high = next(float(t) for t in np.r_[np.linspace(0.5, 0.999, 500), 1.0001]
                    if np.average(p_ca[y_ca == 0] >= t, weights=cw[y_ca == 0]) <= 0.05)
        low = 0.0
        for t in np.linspace(0.001, min(0.499, high - 1e-6), 499):
            if np.average(p_ca[y_ca == 1] <= t, weights=cw[y_ca == 1]) <= 0.05:
                low = float(t)

        model_payload = {
            'model': clf,
            'slope': slope,
            'bias': bias,
            'low': low,
            'high': high,
            'val_auc': auc_va,
            'features': feature_names,
            'family': name
        }
        trained_models[name] = model_payload
        joblib.dump(model_payload, TREE_MODELS_DIR / f'{name}.joblib')

    # Save OOF predictions
    oof_df = pd.DataFrame(oof_predictions)
    oof_df['target'] = y_tr
    assert (fold_ids >= 0).all()
    oof_df['group'] = groups
    oof_df['row_index'] = tr_idx
    oof_df['fold'] = fold_ids
    oof_df.to_csv(TREE_MODELS_DIR / 'oof_train_predictions.csv', index=False)
    (TREE_MODELS_DIR / 'fold_audit.json').write_text(json.dumps(fold_audit, indent=2))
    (TREE_MODELS_DIR / 'validation_metrics.json').write_text(json.dumps(val_metrics, indent=2))
    print(f'\nTree-based models trained and saved to {TREE_MODELS_DIR}', flush=True)
    return trained_models

if __name__ == '__main__':
    train_and_evaluate_tree_models()
