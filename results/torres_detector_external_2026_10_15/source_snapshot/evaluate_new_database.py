"""
Evaluate models on a historical corpus (active main uses the corrected frozen protocol):
Synthetic Swarm Dataset (Swarm_audio_balanced_v2 & Swarm_label_balanced_v2).

315 Multi-mosquito recordings (16 kHz, 5s each, 1575 windows).
Ground truth JSON: exact species counts (Ae. aegypti, Ae. albopictus, An. gambiae,
An. arabiensis, C. pipiens, C. quinquefasciatus).

Evaluates:
- Tree models: LightGBM, HistGradientBoosting, ExtraTrees, Balanced Random Forest
- Deep Neural Network: DeepBioacousticHybridNet
- Committees: Soft-Voting Committee & Stacking Meta-Classifier
- Classical Baselines: SVM RBF, ExtraTrees, GradientBoosting
"""

import os
import json
import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import joblib
from scipy.special import expit
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)

from aedes_classical import raw_scores, decisions, features as extract_base_features
from bioacoustic_invariance import extract_batch_bioacoustic_features, BIOACOUSTIC_FEATURE_NAMES

ROOT = Path('/home/lciarallo/Documentos/Git_projects/mosquito-wingbeat')
AUDIO_DIR = ROOT / 'data/external/additional/synthetic/Swarm_audio_balanced_v2'
LABEL_DIR = ROOT / 'data/external/additional/synthetic/Swarm_label_balanced_v2'
RESULTS_DIR = ROOT / 'results/benchmark_new_database'


SR = 16000
WINDOW_SIZE = 15872  # 0.992 seconds

def load_and_preprocess_dataset():
    print('Loading Synthetic Swarm dataset...', flush=True)
    wav_files = sorted(glob.glob(str(AUDIO_DIR / '*.wav')))
    assert len(wav_files) == 315, f'Expected 315 files, found {len(wav_files)}'

    file_metadata = []
    window_metadata = []
    all_waves = []

    for fpath_str in wav_files:
        fpath = Path(fpath_str)
        stem = fpath.stem
        lpath = LABEL_DIR / f'{stem}.json'
        with open(lpath, 'r', encoding='utf-8') as fp:
            ldata = json.load(fp)

        num_mosq = ldata.get('num_mosquitoes', 0)
        sp_counts = ldata.get('species_count', {})
        aegypti_count = sp_counts.get('Ae. aegypti', 0)
        albopictus_count = sp_counts.get('Ae. albopictus', 0)
        gambiae_count = sp_counts.get('An. gambiae', 0)
        arabiensis_count = sp_counts.get('An. arabiensis', 0)
        pipiens_count = sp_counts.get('C. pipiens', 0)
        quinquefasciatus_count = sp_counts.get('C. quinquefasciatus', 0)

        is_aedes_aegypti = 1 if aegypti_count > 0 else 0

        w, sr = sf.read(fpath, dtype='float32', always_2d=True)
        w = w.mean(axis=1)
        if sr != SR:
            from scipy.signal import resample_poly
            d = np.gcd(sr, SR)
            w = resample_poly(w, SR // d, sr // d)

        n_win = len(w) // WINDOW_SIZE
        file_waves = w[:n_win * WINDOW_SIZE].reshape(n_win, WINDOW_SIZE)
        all_waves.append(file_waves)

        f_meta = {
            'filename': fpath.name,
            'num_mosquitoes': num_mosq,
            'is_aedes_aegypti': is_aedes_aegypti,
            'aegypti_count': aegypti_count,
            'albopictus_count': albopictus_count,
            'gambiae_count': gambiae_count,
            'arabiensis_count': arabiensis_count,
            'pipiens_count': pipiens_count,
            'quinquefasciatus_count': quinquefasciatus_count,
            'num_windows': n_win
        }
        file_metadata.append(f_meta)

        for win_idx in range(n_win):
            w_meta = dict(f_meta)
            w_meta['window_idx'] = win_idx
            window_metadata.append(w_meta)

    waves = np.concatenate(all_waves, axis=0)
    print(f'Extracted {len(waves)} windows from {len(file_metadata)} files.', flush=True)

    # Audio quality check (same as operational pipeline)
    pcm = np.rint(np.clip(waves, -1, 32767 / 32768) * 32768).astype('int16').astype('int32')
    quality = (np.abs(pcm).max(1) >= 32) & ((np.abs(pcm) >= 32700).sum(1) * 100 <= WINDOW_SIZE)

    df_files = pd.DataFrame(file_metadata)
    df_windows = pd.DataFrame(window_metadata)
    df_windows['quality_ok'] = quality

    return waves, df_files, df_windows

def extract_features(waves):
    cache_path = RESULTS_DIR / 'synthetic_swarm_features.npz'
    if cache_path.exists():
        print(f'Loading cached features from {cache_path}...', flush=True)
        data = np.load(cache_path)
        return data['base_features'], data['bio_features'], data['combined_features']

    print('Computing classical (122) and bioacoustic invariance (16) features...', flush=True)
    t0 = time.time()
    base_feats = extract_base_features(waves).astype(np.float32)
    bio_feats = extract_batch_bioacoustic_features(waves, sr=SR).astype(np.float32)
    combined_feats = np.concatenate([base_feats, bio_feats], axis=1).astype(np.float32)
    elapsed = time.time() - t0
    print(f'Feature extraction completed in {elapsed:.2f}s ({elapsed/len(waves):.4f}s/window).', flush=True)

    np.savez_compressed(
        cache_path,
        base_features=base_feats,
        bio_features=bio_feats,
        combined_features=combined_feats
    )
    return base_feats, bio_feats, combined_feats

def compute_metrics(y_true, scores, threshold=0.5):
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = (sens + spec) / 2.0
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, scores)
    except Exception:
        auc = 0.5
        
    return {
        'roc_auc': float(auc),
        'balanced_acc': float(bal_acc),
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'specificity': float(spec),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'threshold': float(threshold)
    }

def legacy_main():
    waves, df_files, df_windows = load_and_preprocess_dataset()
    base_feats, bio_feats, combined_feats = extract_features(waves)

    y_win = df_windows['is_aedes_aegypti'].to_numpy()
    y_file = df_files['is_aedes_aegypti'].to_numpy()
    quality = df_windows['quality_ok'].to_numpy()

    print(f'\nDataset composition:')
    print(f'Files: {len(df_files)} total | {y_file.sum()} Aedes (+) | {(1-y_file).sum()} Non-Aedes (-)')
    print(f'Windows: {len(df_windows)} total | {y_win.sum()} Aedes (+) | {(1-y_win).sum()} Non-Aedes (-)')
    print(f'Audio quality ok: {quality.sum()}/{len(quality)} ({quality.mean()*100:.1f}%)')

    models_predictions_win = {}
    
    # 1. Classical Baselines (122 base features)
    classical_dir = ROOT / 'results/aedes_classical/models'
    for c_name in ['svm_rbf', 'extra_trees', 'gradient_boosting']:
        c_path = classical_dir / f'{c_name}.joblib'
        if not c_path.exists():
            continue
        c_dict = joblib.load(c_path)
        m = c_dict['model']
        sl, bi = c_dict['slope'], c_dict['bias']
        z = raw_scores(m, base_feats)
        p = expit(sl * z + bi)
        models_predictions_win[f'baseline_{c_name}'] = p

    # 2. Tree Models (138 combined features)
    tree_dir = ROOT / 'results/models_trees'
    for t_name in ['lightgbm', 'hist_gradient_boosting', 'extra_trees', 'random_forest']:
        t_path = tree_dir / f'{t_name}.joblib'
        if not t_path.exists():
            continue
        t_dict = joblib.load(t_path)
        m = t_dict['model']
        sl, bi = t_dict['slope'], t_dict['bias']
        z = raw_scores(m, combined_feats)
        p = expit(sl * z + bi)
        models_predictions_win[f'tree_{t_name}'] = p

    # 3. Deep Hybrid Net (Audio waveforms + 138 standardized features)
    deep_path = ROOT / 'results/models_deep/deep_hybrid_net.pt'
    if deep_path.exists():
        import torch
        from models_deep import DeepBioacousticHybridNet
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Deep model using device: {device}', flush=True)
        saved_deep = torch.load(deep_path, map_location=device, weights_only=False)
        deep_model = DeepBioacousticHybridNet(bio_features_dim=combined_feats.shape[1]).to(device)
        deep_model.load_state_dict(saved_deep['state_dict'])
        deep_model.eval()

        mean_sc = saved_deep['scaler_mean'].astype(np.float32)
        scale_sc = saved_deep['scaler_scale'].astype(np.float32)
        d_slope = saved_deep['slope']
        d_bias = saved_deep['bias']

        b_std = ((combined_feats - mean_sc) / scale_sc).astype(np.float32)
        w_t = torch.from_numpy(waves.astype(np.float32)).to(device)
        b_t = torch.from_numpy(b_std).to(device)
        with torch.no_grad():
            deep_logits = deep_model(w_t, b_t, augment=False).cpu().numpy()
        p_deep = expit(d_slope * deep_logits + d_bias)
        models_predictions_win['deep_hybrid_net'] = p_deep

    # 4. Committee Models
    comm_path = ROOT / 'results/models_committee/committees_bundle.joblib'
    if comm_path.exists():
        cb = joblib.load(comm_path)
        soft_comm = cb['soft_voting_committee']
        stack_comm = cb['stacking_committee']

        p_soft = soft_comm.predict_proba(combined_feats)
        p_stack = stack_comm.predict_proba(combined_feats)
        models_predictions_win['committee_soft_voting'] = p_soft
        models_predictions_win['committee_stacking'] = p_stack

    # Save predictions
    df_win_preds = df_windows.copy()
    for m_name, p in models_predictions_win.items():
        df_win_preds[m_name] = p
    df_win_preds.to_csv(RESULTS_DIR / 'synthetic_swarm_window_predictions.csv', index=False)

    # Compute metrics for each model
    results_list = []
    print('\n' + '='*80)
    print('EVALUATION RESULTS ON COMPLETELY NEW DATABASE: SYNTHETIC SWARM')
    print('='*80)

    for m_name, p_win in models_predictions_win.items():
        # Window-level metrics (default threshold 0.5)
        m_win = compute_metrics(y_win, p_win, threshold=0.5)
        
        # Recording-level metrics (aggregation by mean probability across windows of file)
        df_win_preds['pred'] = p_win
        p_file_mean = df_win_preds.groupby('filename')['pred'].mean().to_numpy()
        p_file_max = df_win_preds.groupby('filename')['pred'].max().to_numpy()

        m_file_mean = compute_metrics(y_file, p_file_mean, threshold=0.5)
        m_file_max = compute_metrics(y_file, p_file_max, threshold=0.5)

        results_list.append({
            'model': m_name,
            'win_auc': m_win['roc_auc'],
            'win_bal_acc': m_win['balanced_acc'],
            'win_acc': m_win['accuracy'],
            'win_prec': m_win['precision'],
            'win_rec': m_win['recall'],
            'win_spec': m_win['specificity'],
            'file_mean_auc': m_file_mean['roc_auc'],
            'file_mean_bal_acc': m_file_mean['balanced_acc'],
            'file_mean_acc': m_file_mean['accuracy'],
            'file_max_bal_acc': m_file_max['balanced_acc'],
            'file_max_auc': m_file_max['roc_auc']
        })

    res_df = pd.DataFrame(results_list)
    res_df.to_csv(RESULTS_DIR / 'synthetic_swarm_model_metrics.csv', index=False)
    
    # Pretty print summary table
    print(res_df.to_string(index=False, formatters={
        'win_auc': '{:.4f}'.format,
        'win_bal_acc': '{:.2%}'.format,
        'win_acc': '{:.2%}'.format,
        'win_prec': '{:.2%}'.format,
        'win_rec': '{:.2%}'.format,
        'win_spec': '{:.2%}'.format,
        'file_mean_auc': '{:.4f}'.format,
        'file_mean_bal_acc': '{:.2%}'.format,
        'file_mean_acc': '{:.2%}'.format,
        'file_max_bal_acc': '{:.2%}'.format,
        'file_max_auc': '{:.4f}'.format,
    }))

    # Detailed inspection: species-by-species false positive breakdown for top models
    print('\n' + '-'*80)
    print('FALSE POSITIVE ANALYSIS ACROSS NON-AEDES SWARMS (Purity breakdown):')
    print('-'*80)
    
    non_aedes_mask = (y_file == 0)
    non_aedes_files = df_files[non_aedes_mask].copy()
    
    # Map predominant species in non-aedes swarms
    def get_predominant(row):
        sp_map = {
            'albopictus': row['albopictus_count'],
            'gambiae': row['gambiae_count'],
            'arabiensis': row['arabiensis_count'],
            'pipiens': row['pipiens_count'],
            'quinquefasciatus': row['quinquefasciatus_count']
        }
        return max(sp_map, key=sp_map.get)

    non_aedes_files['dominant_species'] = non_aedes_files.apply(get_predominant, axis=1)

    for m_name in ['baseline_extra_trees', 'tree_extra_trees', 'tree_lightgbm', 'committee_soft_voting', 'committee_stacking']:
        if m_name not in models_predictions_win:
            continue
        p_mean = df_win_preds.groupby('filename')[m_name].mean()
        non_aedes_files[f'p_{m_name}'] = non_aedes_files['filename'].map(p_mean)
        fpr_overall = (non_aedes_files[f'p_{m_name}'] >= 0.5).mean()
        print(f'{m_name:25s} | Non-Aedes FPR @ 0.5: {fpr_overall:.2%}')
        for sp, grp in non_aedes_files.groupby('dominant_species'):
            sp_fpr = (grp[f'p_{m_name}'] >= 0.5).mean()
            print(f'   - Dominant {sp:18s} (N={len(grp):2d}): FPR = {sp_fpr:.2%}')

def main():
    # Historical probability/feature helpers remain available for reproduction.
    # Active evaluation uses the frozen corrected protocol and writes separate results.
    from corrected_validation import evaluate
    return evaluate()

if __name__ == '__main__':
    main()
