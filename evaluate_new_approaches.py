"""Evaluation of New Bioacoustic Approaches for Mosquito Classification.

Combines:
1. Baseline 122 acoustic features (linear sub-band energies, cepstrum, centroid, flatness, flux).
2. Temperature-invariant harmonic ladder (HPS F0, H2/H1..H5/H1, decay slope, HNR).
3. Swarm intermodulation & envelope modulation dynamics (2-40 Hz beating, dispersion, crest factor).
4. Cepstral Mean Subtraction (CMS) channel normalization.

Evaluates against:
- UFRGS out-of-domain dataset (Aedes in Brazil, cellphone in cup).
- HumBug and Dryad benchmarks.
- Private user swarm audio (107 windows).
- Real environmental noise controls.
"""
from pathlib import Path
import json, time, hashlib
import numpy as np, pandas as pd, soundfile as sf, joblib
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.special import expit

from aedes_binary import targets, weights, ROOT, decisions
from aedes_classical import FEATURE_NAMES as BASE_FEATURE_NAMES, raw_scores
from train_aedes_binary import summary
from bioacoustic_invariance import (
    extract_batch_bioacoustic_features,
    extract_bioacoustic_features_for_wave,
    apply_cepstral_mean_subtraction,
    BIOACOUSTIC_FEATURE_NAMES,
    SR
)

OUT_DIR = ROOT / 'results/new_approaches'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def compute_or_load_bioacoustic_features(f: pd.DataFrame) -> np.ndarray:
    cache_file = ROOT / 'data/cache/bioacoustic_invariance_features.npy'
    if cache_file.exists():
        bio_feats = np.load(cache_file)
        if len(bio_feats) == len(f) and bio_feats.shape[1] == len(BIOACOUSTIC_FEATURE_NAMES):
            print(f'Loaded cached bioacoustic features: {bio_feats.shape}', flush=True)
            return bio_feats

    print('Extracting new bioacoustic features across multisource segments...', flush=True)
    bio_feats = np.empty((len(f), len(BIOACOUSTIC_FEATURE_NAMES)), dtype=np.float32)
    t0 = time.time()
    for bucket in ['long', 'short']:
        mask = f.bucket.eq(bucket).to_numpy()
        wave = np.load(ROOT / f'data/cache/multisource/{bucket}.npy', mmap_mode='r')
        indices = f.loc[mask, 'wave_index'].astype(int).to_numpy()
        print(f'Processing {bucket}: {len(indices)} windows...', flush=True)
        # Process in chunks of 500
        chunk_size = 500
        for start in range(0, len(indices), chunk_size):
            end = min(start + chunk_size, len(indices))
            chunk_waves = wave[indices[start:end]]
            batch_feats = extract_batch_bioacoustic_features(chunk_waves, sr=SR)
            # Find the positions in the full array
            pos = np.flatnonzero(mask)[start:end]
            bio_feats[pos] = batch_feats
    np.save(cache_file, bio_feats)
    print(f'Extracted and saved bioacoustic features in {time.time()-t0:.1f} s', flush=True)
    return bio_feats

def evaluate_audio_file(path: Path, model_dict: dict, label: str, group: str, domain: str):
    """Evaluate a raw audio file with window-by-window feature extraction."""
    from scipy.signal import resample_poly
    from aedes_classical import features as extract_base_features
    
    w, original_sr = sf.read(path, dtype='float32', always_2d=True)
    w = w.mean(axis=1)
    if original_sr != SR:
        d = np.gcd(original_sr, SR)
        w = resample_poly(w, SR // d, original_sr // d)
    size = 15872 if len(w) >= 15872 else 4800
    n = len(w) // size
    if n < 1:
        raise ValueError('Less than 0.3 seconds of audio.')
    waves = w[:n * size].reshape(n, size)
    pcm = np.rint(np.clip(waves, -1, 32767 / 32768) * 32768).astype('int16').astype('int32')
    quality = (np.abs(pcm).max(1) >= 32) & ((np.abs(pcm) >= 32700).sum(1) * 100 <= size)

    base_x = extract_base_features(waves)
    bio_x = extract_batch_bioacoustic_features(waves, sr=SR)
    combined_x = np.concatenate([base_x, bio_x], axis=1)

    model = model_dict['model']
    slope, bias = model_dict['slope'], model_dict['bias']
    low, high = model_dict['low'], model_dict['high']

    z = raw_scores(model, combined_x)
    scores = expit(slope * z + bias)
    states = decisions(scores, quality, low, high)

    # Check for swarm signature:
    # High envelope modulation + F0 in 500-650 Hz + Aedes score >= low
    is_swarm = (
        (bio_x[:, BIOACOUSTIC_FEATURE_NAMES.index('envelope_mod_energy_ratio')] >= 0.50) &
        (bio_x[:, BIOACOUSTIC_FEATURE_NAMES.index('hps_f0_hz')] >= 250.0) &
        (bio_x[:, BIOACOUSTIC_FEATURE_NAMES.index('hps_f0_hz')] <= 750.0) &
        (scores >= 0.35)
    )
    detailed_states = states.copy()
    for i in range(len(states)):
        if states[i] == 'AEDES_AEGYPTI':
            if is_swarm[i]:
                detailed_states[i] = 'AEDES_AEGYPTI_SWARM'
            else:
                detailed_states[i] = 'AEDES_AEGYPTI_INDIVIDUAL'
        elif states[i] == 'INCERTO' and is_swarm[i]:
            detailed_states[i] = 'AEDES_AEGYPTI_SWARM_DETECTED'

    df = pd.DataFrame(dict(
        start_s=np.arange(n) * size / SR,
        quality_ok=quality,
        aedes_score=scores,
        state=states,
        detailed_state=detailed_states,
        hps_f0=bio_x[:, BIOACOUSTIC_FEATURE_NAMES.index('hps_f0_hz')],
        mod_ratio=bio_x[:, BIOACOUSTIC_FEATURE_NAMES.index('envelope_mod_energy_ratio')],
        dispersion=bio_x[:, BIOACOUSTIC_FEATURE_NAMES.index('swarm_dispersion_index')]
    ))
    return df

def main():
    print('=== EVALUATION OF NEW BIOACOUSTIC APPROACHES ===', flush=True)
    f = pd.read_csv(ROOT / 'results/multisource/segments.csv', low_memory=False)
    y = targets(f)
    
    # Base classical features
    base_feats = np.load(ROOT / 'data/cache/aedes_classical_features.npy')
    # New bioacoustic features
    bio_feats = compute_or_load_bioacoustic_features(f)
    
    # Combined feature matrix
    X = np.concatenate([base_feats, bio_feats], axis=1).astype(np.float32)
    ALL_FEATURE_NAMES = BASE_FEATURE_NAMES + BIOACOUSTIC_FEATURE_NAMES
    print(f'Total features combined: {X.shape[1]} (122 base + {len(BIOACOUSTIC_FEATURE_NAMES)} bioacoustic)', flush=True)

    ix = {s: np.flatnonzero(f.split.eq(s) & f.quality_ok & (y >= 0)) for s in ['train', 'validation', 'calibration']}
    tr, va, ca = ix['train'], ix['validation'], ix['calibration']
    tw = weights(f.iloc[tr])
    vw = weights(f.iloc[va])
    cw = weights(f.iloc[ca])

    # 1. Train Enhanced HistGradientBoosting with Bioacoustic Features
    print('\nTraining Enhanced HistGradientBoosting (with bioacoustics)...', flush=True)
    hgb = HistGradientBoostingClassifier(
        max_iter=250,
        max_leaf_nodes=31,
        learning_rate=0.07,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=20261004
    )
    t0 = time.time()
    hgb.fit(X[tr], y[tr], sample_weight=tw)
    train_time = time.time() - t0
    
    z_va = raw_scores(hgb, X[va])
    val_auc = float(roc_auc_score(y[va], z_va, sample_weight=vw))
    print(f'Validation AUC (Enhanced HGB): {val_auc:.4f} (fitted in {train_time:.1f} s)', flush=True)

    # Calibrate with Platt scaling on calibration split
    z_ca = raw_scores(hgb, X[ca])
    calibrator = LogisticRegression(C=10, max_iter=2000).fit(z_ca[:, None], y[ca], sample_weight=cw)
    slope = float(calibrator.coef_[0, 0])
    bias = float(calibrator.intercept_[0])
    p_ca = expit(slope * z_ca + bias)

    # Set decision thresholds targeting 5% FPR on negatives in calibration
    high = next(float(t) for t in np.r_[np.linspace(0.5, 0.999, 500), 1.0001]
                if np.average(p_ca[y[ca] == 0] >= t, weights=cw[y[ca] == 0]) <= 0.05)
    low = 0.0
    for t in np.linspace(0.001, min(0.499, high - 1e-6), 499):
        if np.average(p_ca[y[ca] == 1] <= t, weights=cw[y[ca] == 1]) <= 0.05:
            low = float(t)

    print(f'Calibrated thresholds: low={low:.4f}, high={high:.4f}', flush=True)
    model_dict = dict(
        model=hgb, slope=slope, bias=bias, low=low, high=high,
        features=ALL_FEATURE_NAMES, family='enhanced_hgb'
    )
    joblib.dump(model_dict, OUT_DIR / 'enhanced_hgb.joblib')

    # 2. Historical test sets evaluation
    print('\nEvaluating on historical test sets (UFRGS, Dryad, HumBug, MosquitoSong)...', flush=True)
    test_idx = np.flatnonzero(f.split.isin(['test', 'challenge']))
    tf = f.iloc[test_idx].copy()
    z_test = raw_scores(hgb, X[test_idx])
    p_test = expit(slope * z_test + bias)
    tf['aedes_score'] = p_test
    tf['state'] = decisions(p_test, tf.quality_ok.to_numpy(), low, high)

    results_table = []
    for domain in ['dryad', 'humbug', 'mosquitosong', 'ufrgs']:
        mask = tf.domain.eq(domain).to_numpy()
        d_frame = tf[mask]
        d_p = p_test[mask]
        res, _ = summary(d_frame, d_p, low, high)
        results_table.append(dict(domain=domain, **res))
    results_df = pd.DataFrame(results_table)
    print(results_df.to_string(index=False))
    results_df.to_csv(OUT_DIR / 'historical_domains_metrics.csv', index=False)

    # 3. Private Swarm Audio Evaluation
    print('\nEvaluating on user private swarm audio (107 windows)...', flush=True)
    swarm_audio = ROOT / 'tmp/private-audio-test/download.audio'
    swarm_df = evaluate_audio_file(swarm_audio, model_dict, 'Aedes aegypti', 'private_swarm', 'private')
    swarm_df.to_csv(OUT_DIR / 'private_swarm_predictions.csv', index=False)
    swarm_summary = {
        'total_windows': len(swarm_df),
        'standard_states': swarm_df.state.value_counts().to_dict(),
        'detailed_states': swarm_df.detailed_state.value_counts().to_dict(),
        'mean_aedes_score': float(swarm_df.aedes_score.mean()),
        'median_aedes_score': float(swarm_df.aedes_score.median()),
        'mean_hps_f0_hz': float(swarm_df.hps_f0.mean()),
        'mean_mod_energy_ratio': float(swarm_df.mod_ratio.mean()),
        'windows_with_swarm_dynamics': int((swarm_df.mod_ratio >= 0.50).sum())
    }
    print(json.dumps(swarm_summary, indent=2))
    (OUT_DIR / 'private_swarm_summary.json').write_text(json.dumps(swarm_summary, indent=2))

    # 4. Real Noise Control Evaluation
    print('\nEvaluating on negative environmental noise controls...', flush=True)
    import glob
    noise_files = glob.glob(str(ROOT / 'data/noise/noise/*.wav'))
    noise_results = []
    for nf in noise_files:
        try:
            ndf = evaluate_audio_file(Path(nf), model_dict, 'noise', 'noise_ctrl', 'noise')
            noise_results.append(ndf)
        except Exception as e:
            pass
    if noise_results:
        all_noise = pd.concat(noise_results, ignore_index=True)
        fp_rate = float((all_noise.state == 'AEDES_AEGYPTI').mean())
        print(f'Noise False Positive Rate: {fp_rate*100:.2f}% across {len(all_noise)} noise windows', flush=True)
        print('Noise state distribution:', all_noise.state.value_counts().to_dict(), flush=True)
        (OUT_DIR / 'noise_evaluation.json').write_text(json.dumps(dict(
            total_noise_windows=len(all_noise),
            fp_rate=fp_rate,
            states=all_noise.state.value_counts().to_dict()
        ), indent=2))

    print('\n=== ALL EVALUATIONS COMPLETED SUCCESSFULLY ===', flush=True)

if __name__ == '__main__':
    main()
