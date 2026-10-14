"""
Evaluate models on a second historical corpus (active main uses the corrected frozen protocol):
BioDCASE Challenge Domain D5 (Optical sensor / distinct modality).
Domain D5 was completely excluded from multisource training and earlier evaluations.

Extracts a balanced test sample of 400 recordings:
- 200 Aedes aegypti (S_1_D_5_*)
- 200 Non-Aedes (70 Culex quinquefasciatus S_3, 70 Anopheles gambiae S_4, 60 Culex pipiens S_7)
"""

import io
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import joblib
from scipy.signal import resample_poly
from scipy.special import expit
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)

from aedes_classical import raw_scores, features as extract_base_features
from bioacoustic_invariance import extract_batch_bioacoustic_features

ROOT = Path('/home/lciarallo/Documentos/Git_projects/mosquito-wingbeat')
ZIP_PATH = ROOT / 'data/external/additional/Development_data.zip'
RESULTS_DIR = ROOT / 'results/benchmark_new_database'


SR = 16000
WINDOW_SIZE = 4800  # 0.3s (short format standard for D5 recordings)

def extract_and_process_d5(sample_n_pos=200, sample_n_neg=200):
    print('Reading BioDCASE D5 files directly from zip...', flush=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        names = zf.namelist()
        s1_files = sorted([n for n in names if 'raw_audio/S_1_D_5_' in n and n.endswith('.wav') and not n.startswith('__MACOSX')])[:sample_n_pos]
        s3_files = sorted([n for n in names if 'raw_audio/S_3_D_5_' in n and n.endswith('.wav') and not n.startswith('__MACOSX')])[:70]
        s4_files = sorted([n for n in names if 'raw_audio/S_4_D_5_' in n and n.endswith('.wav') and not n.startswith('__MACOSX')])[:70]
        s7_files = sorted([n for n in names if 'raw_audio/S_7_D_5_' in n and n.endswith('.wav') and not n.startswith('__MACOSX')])[:60]

        selected_files = [(f, 1, 'Ae. aegypti') for f in s1_files] + \
                         [(f, 0, 'C. quinquefasciatus') for f in s3_files] + \
                         [(f, 0, 'An. gambiae') for f in s4_files] + \
                         [(f, 0, 'C. pipiens') for f in s7_files]

        all_waves = []
        meta = []
        for fpath, label, sp in selected_files:
            data = zf.read(fpath)
            w, orig_sr = sf.read(io.BytesIO(data), dtype='float32')
            if orig_sr != SR:
                d = np.gcd(orig_sr, SR)
                w = resample_poly(w, SR // d, orig_sr // d)
            if len(w) < WINDOW_SIZE:
                continue
            crop_w = w[:WINDOW_SIZE]
            all_waves.append(crop_w)
            meta.append({'file': Path(fpath).name, 'is_aedes_aegypti': label, 'species': sp})

    waves = np.array(all_waves, dtype=np.float32)
    df_meta = pd.DataFrame(meta)
    print(f'Extracted {len(waves)} recordings from BioDCASE Domain D5.', flush=True)
    return waves, df_meta

def legacy_main():
    waves, df_meta = extract_and_process_d5()
    print('Computing features on D5...', flush=True)
    t0 = time.time()
    base_feats = extract_base_features(waves).astype(np.float32)
    bio_feats = extract_batch_bioacoustic_features(waves, sr=SR).astype(np.float32)
    comb_feats = np.concatenate([base_feats, bio_feats], axis=1).astype(np.float32)
    print(f'Features extracted in {time.time() - t0:.2f}s.', flush=True)

    y = df_meta['is_aedes_aegypti'].to_numpy()

    # Load models
    models_preds = {}

    # 1. Classical Baselines
    for c_name in ['svm_rbf', 'extra_trees', 'gradient_boosting']:
        p_path = ROOT / f'results/aedes_classical/models/{c_name}.joblib'
        if p_path.exists():
            c_dict = joblib.load(p_path)
            z = raw_scores(c_dict['model'], base_feats)
            models_preds[f'baseline_{c_name}'] = expit(c_dict['slope'] * z + c_dict['bias'])

    # 2. Tree Models
    for t_name in ['lightgbm', 'hist_gradient_boosting', 'extra_trees', 'random_forest']:
        t_path = ROOT / f'results/models_trees/{t_name}.joblib'
        if t_path.exists():
            t_dict = joblib.load(t_path)
            z = raw_scores(t_dict['model'], comb_feats)
            models_preds[f'tree_{t_name}'] = expit(t_dict['slope'] * z + t_dict['bias'])

    # 3. Committees
    comm_path = ROOT / 'results/models_committee/committees_bundle.joblib'
    if comm_path.exists():
        cb = joblib.load(comm_path)
        models_preds['committee_soft_voting'] = cb['soft_voting_committee'].predict_proba(comb_feats)
        models_preds['committee_stacking'] = cb['stacking_committee'].predict_proba(comb_feats)

    # 4. Deep Hybrid Net
    deep_path = ROOT / 'results/models_deep/deep_hybrid_net.pt'
    if deep_path.exists():
        import torch
        from models_deep import DeepBioacousticHybridNet
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        saved_deep = torch.load(deep_path, map_location=device, weights_only=False)
        deep_model = DeepBioacousticHybridNet(bio_features_dim=comb_feats.shape[1]).to(device)
        deep_model.load_state_dict(saved_deep['state_dict'])
        deep_model.eval()

        mean_sc = saved_deep['scaler_mean'].astype(np.float32)
        scale_sc = saved_deep['scaler_scale'].astype(np.float32)
        b_std = ((comb_feats - mean_sc) / scale_sc).astype(np.float32)

        # For 4800 sample window, pad/tile to SAMPLES (15872) for deep model forward pass
        repeats = int(np.ceil(15872 / waves.shape[1]))
        waves_padded = np.tile(waves, (1, repeats))[:, :15872].astype(np.float32)

        w_t = torch.from_numpy(waves_padded).to(device)
        b_t = torch.from_numpy(b_std).to(device)
        with torch.no_grad():
            deep_logits = deep_model(w_t, b_t, augment=False).cpu().numpy()
        models_preds['deep_hybrid_net'] = expit(saved_deep['slope'] * deep_logits + saved_deep['bias'])

    print('\n' + '='*80)
    print('EVALUATION RESULTS ON BIODCASE DOMAIN D5 (400 UNSEEN OPTICAL RECORDINGS)')
    print('='*80)

    rows = []
    for m_name, p in models_preds.items():
        auc = roc_auc_score(y, p)
        y_pred = (p >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        bal_acc = (sens + spec) / 2.0
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)

        rows.append({
            'model': m_name,
            'auc': auc,
            'balanced_acc': bal_acc,
            'accuracy': acc,
            'sensitivity': sens,
            'specificity': spec,
            'precision': prec
        })

    df_res = pd.DataFrame(rows)
    df_res.to_csv(RESULTS_DIR / 'biodcase_d5_metrics.csv', index=False)
    print(df_res.to_string(index=False, formatters={
        'auc': '{:.4f}'.format,
        'balanced_acc': '{:.2%}'.format,
        'accuracy': '{:.2%}'.format,
        'sensitivity': '{:.2%}'.format,
        'specificity': '{:.2%}'.format,
        'precision': '{:.2%}'.format,
    }))

def main():
    # Historical probability/feature helpers remain available for reproduction.
    # Active evaluation uses the frozen corrected protocol and writes separate results.
    from corrected_validation import evaluate
    return evaluate()

if __name__ == '__main__':
    main()
