"""Offline, strictly per-one-second-window features; no fitted batch statistics.

SHS is a descriptor, not an entomological frequency lookup. Multiple peaks and
octaves are retained as evidence, rather than treated as independent insects.
"""
import numpy as np
from scipy import signal
from scipy.fft import dct
from scipy.ndimage import median_filter

SR = 16000
PSD_GRID = np.arange(150, 3951, 50)
F0_GRID = np.arange(200, 901, 5)
CONTRAST_GRID = np.arange(200, 3901, 50)
FEATURE_NAMES = (
    [f'psd_invariant_{f}' for f in PSD_GRID]
    + [f'contrast_mean_{f}' for f in CONTRAST_GRID]
    + [f'contrast_std_{f}' for f in CONTRAST_GRID]
    + [f'mfcc_attenuated_mean_{i}' for i in range(13)]
    + [f'mfcc_std_{i}' for i in range(13)]
    + [f'mfcc_delta_abs_{i}' for i in range(13)]
    + [f'mfcc_delta_std_{i}' for i in range(13)]
    + [f'shs_peak{k}_{v}' for k in range(3) for v in ['f0', 'strength', 'h2_h1', 'h3_h1']]
    + ['shs_f0_median', 'shs_f0_std', 'shs_f0_q10', 'shs_f0_q90',
       'shs_profile_entropy', 'shs_peak_energy_fraction', 'shs_peak_spread',
       'shs_top2_ratio', 'spectral_flatness', 'spectral_centroid',
       'spectral_bandwidth', 'harmonic_salience']
)


def _mel_bank(f):
    to_mel = lambda x: 2595 * np.log10(1 + x / 700)
    limits = 700 * (10 ** (np.linspace(to_mel(100), to_mel(3900), 34) / 2595) - 1)
    bank = np.maximum(0, np.minimum(
        (f[None, :] - limits[:-2, None]) / np.diff(limits)[:-1, None],
        (limits[2:, None] - f[None, :]) / np.diff(limits)[1:, None]))
    return bank / np.maximum(bank.sum(1, keepdims=True), 1e-12)


def _comb(f, power):
    """Harmonic contrast vs local median, including a fundamental-presence term."""
    floor = median_filter(power, size=(21, 1), mode='nearest')
    ratio = np.clip(power / np.maximum(floor, 1e-12), 0, 50)
    evidence = np.log1p(np.maximum(ratio - 1, 0))
    components = np.stack([
        np.stack([np.interp(F0_GRID * h, f, evidence[:, t])
                  for t in range(power.shape[1])], axis=1)
        for h in (1, 2, 3, 4)
    ])
    # Penalise subharmonics with no measured fundamental; avoids unconditional
    # downward octave jumps while preserving multi-peak evidence.
    score = np.maximum(sum(components[h-1] / h for h in (1, 2, 3, 4))
                       - .5 * np.maximum(components[1] - components[0], 0), 0)
    return score, components


def window_features(raw_wave):
    """Return (feature vector, bounded salience), using exactly one 1 s window."""
    w = np.asarray(raw_wave, dtype=np.float64)
    if w.ndim != 1 or len(w) != SR or not np.isfinite(w).all():
        raise ValueError('Expected one finite mono 16000-sample window; no padding.')
    w = w - w.mean()
    w = w / max(float(np.sqrt(np.mean(w*w))), 1e-8)
    f, _, z = signal.stft(w, fs=SR, nperseg=3200, noverlap=2880,
                          boundary=None, padded=False)
    power = np.abs(z)**2
    mean_power = power.mean(1)
    logp = 10 * np.log10(np.maximum(power, 1e-12))
    local = median_filter(logp, size=(21, 1), mode='nearest')
    contrast = logp - local
    psd = np.interp(PSD_GRID, f, logp.mean(1))
    # Remove gain and static log-frequency tilt within this window alone.
    axis = np.log(PSD_GRID.astype(float))
    axis -= axis.mean()
    centred = psd - np.median(psd)
    centred -= axis * np.dot(axis, centred) / np.dot(axis, axis)
    centred -= np.median(centred)
    scale = max(float(np.subtract(*np.quantile(centred, [.75, .25]))), 1.)
    psd_invariant = np.clip(centred / scale, -8, 8)
    mel_db = 10 * np.log10(np.maximum(_mel_bank(f) @ power, 1e-12))
    mel_db -= np.median(mel_db, axis=0, keepdims=True)
    mfcc = dct(mel_db, type=2, axis=0, norm='ortho')[:13]
    means = mfcc.mean(1)
    means[:3] = 0.  # Fixed predeclared suppression of static low-order channel offsets.
    means[3] *= .25
    delta = np.diff(mfcc, axis=1)
    shs, harmonics = _comb(f, power)
    profile = shs.mean(1)
    peaks, _ = signal.find_peaks(profile, distance=5)
    if not len(peaks):
        peaks = np.array([int(profile.argmax())])
    ranked = sorted(peaks, key=lambda i: (-profile[i], int(i)))[:3]
    peak_features = []
    for i in ranked:
        h = harmonics[:, i].mean(1)
        peak_features.extend([F0_GRID[i], profile[i], h[1]/max(h[0], .1), h[2]/max(h[0], .1)])
    peak_features.extend([0.] * (12-len(peak_features)))
    track = F0_GRID[shs.argmax(0)]
    q = profile / max(float(profile.sum()), 1e-12)
    band = (f >= 150) & (f <= 3900)
    spectral = mean_power[band] / max(float(mean_power[band].sum()), 1e-12)
    centroid = float(np.dot(f[band], spectral))
    bandwidth = float(np.sqrt(np.dot((f[band]-centroid)**2, spectral)))
    mask = np.zeros(len(f), bool)
    for i in ranked:
        for h in (1, 2, 3, 4):
            mask |= np.abs(f - F0_GRID[i]*h) <= 15
    fraction = float(mean_power[mask & band].sum()/max(mean_power[band].sum(), 1e-12))
    # Noise retains a floor, so no group is silently deleted by salience.
    salience = float(np.clip(.1 + profile.max()/3 * np.sqrt(fraction), .1, 3.))
    summary = [np.median(track), np.std(track), *np.quantile(track, [.1, .9]),
               -np.sum(q*np.log(np.maximum(q, 1e-12)))/np.log(len(q)), fraction,
               np.sqrt(np.dot((F0_GRID-np.dot(F0_GRID,q))**2,q)),
               profile[ranked[1]]/max(profile[ranked[0]], .1) if len(ranked)>1 else 0.,
               np.exp(np.log(np.maximum(spectral,1e-12)).mean())/max(spectral.mean(),1e-12),
               centroid, bandwidth, salience]
    features = np.r_[psd_invariant,
                     np.interp(CONTRAST_GRID,f,contrast.mean(1)),
                     np.interp(CONTRAST_GRID,f,contrast.std(1)),
                     means, mfcc.std(1), np.abs(delta).mean(1), delta.std(1),
                     peak_features, summary].astype(np.float32)
    if len(features) != len(FEATURE_NAMES) or not np.isfinite(features).all():
        raise ValueError('Nonfinite or misaligned features.')
    return features, salience


def extract_features(windows):
    """Deliberately row-wise: adding/changing any other window changes no bits."""
    items = [window_features(w) for w in windows]
    return np.stack([x[0] for x in items]), np.array([x[1] for x in items])
