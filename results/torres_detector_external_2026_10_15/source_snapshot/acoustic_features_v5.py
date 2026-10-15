"""V5 channel-invariant inductive frontend: bandpass-conditioned 250-3150 Hz grid,
log-frequency tilt removal, Rayleigh-corrected SHS comb, and sub-band modulation.

Operates strictly within a single 1-second (16000-sample) window with zero cross-window
or batch state. Pre-conditioning suppresses <220 Hz room rumble, 60 Hz mains harmonics
(180..360 Hz), and >3250 Hz Nyquist transition artifacts so 8 kHz and 16 kHz recordings
share an identical passband.
"""
import numpy as np
from scipy import signal
from scipy.fft import dct, rfft
from scipy.ndimage import median_filter, maximum_filter1d

SR = 16000
SHORT_NPERSEG, SHORT_HOP = 1600, 160
LONG_NPERSEG, LONG_HOP = 3200, 320
NOISE_MEDIAN_HZ = 150
LN2 = float(np.log(2.0))
PSD_GRID = np.arange(250, 3151, 50)
CONTRAST_GRID = np.arange(250, 3151, 50)
F0_GRID = np.arange(310, 751, 5)
FEATURE_NAMES = (
    [f'psd_local_snr_{f}' for f in PSD_GRID]
    + [f'contrast_short_mean_{f}' for f in CONTRAST_GRID]
    + [f'contrast_short_std_{f}' for f in CONTRAST_GRID]
    + [f'mfcc_centred_mean_{i}' for i in range(13)]
    + [f'mfcc_short_std_{i}' for i in range(13)]
    + [f'mfcc_short_delta_abs_{i}' for i in range(13)]
    + [f'mfcc_short_delta_std_{i}' for i in range(13)]
    + [f'shs_long_peak{k}_{v}' for k in range(3) for v in ['f0', 'strength', 'energy_h2_h1', 'energy_h3_h1']]
    + ['shs_short_f0_median', 'shs_short_f0_std', 'shs_short_f0_q10', 'shs_short_f0_q90',
       'shs_long_entropy', 'harmonic_energy_fraction', 'shs_long_spread',
       'shs_long_top2_ratio', 'spectral_flatness', 'spectral_centroid',
       'spectral_bandwidth', 'harmonic_salience',
       'energy_500_1600_fraction', 'energy_500_1600_vs_250_500',
       'shs_long_f0_median', 'shs_long_f0_std', 'shs_short_long_disagreement',
       'low_peak_ratio_mean', 'low_peak_ratio_std', 'low_peak_freq_mean', 'low_peak_freq_std',
       'subband_mod_low', 'subband_mod_mid', 'subband_mod_high']
)

_SOS_HP = signal.butter(4, 220.0 / (SR / 2.0), btype='highpass', output='sos')
_SOS_LP = signal.butter(4, 3250.0 / (SR / 2.0), btype='lowpass', output='sos')
_NOTCHES = [signal.iirnotch(f0, Q=30.0, fs=SR) for f0 in (180.0, 240.0, 300.0, 360.0)]
_LOG_AXIS = np.log(PSD_GRID.astype(np.float64))
_LOG_AXIS -= _LOG_AXIS.mean()
_LOG_AXIS_NORM = float(np.dot(_LOG_AXIS, _LOG_AXIS))
_HANN512 = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(512) / 511.0)).astype(np.float64)


def _mel_bank(f):
    mel = lambda x: 2595.0 * np.log10(1.0 + x / 700.0)
    bounds = 700.0 * (10.0 ** (np.linspace(mel(220.0), mel(3150.0), 34) / 2595.0) - 1.0)
    bank = np.maximum(0.0, np.minimum(
        (f[None, :] - bounds[:-2, None]) / np.diff(bounds)[:-1, None],
        (bounds[2:, None] - f[None, :]) / np.diff(bounds)[1:, None]))
    return bank / np.maximum(bank.sum(1, keepdims=True), 1e-12)


SHORT_F = np.fft.rfftfreq(SHORT_NPERSEG, 1.0 / SR)
LONG_F = np.fft.rfftfreq(LONG_NPERSEG, 1.0 / SR)
MEL_BANK_SHORT = _mel_bank(SHORT_F)


def precondition_window(raw_wave):
    w = np.asarray(raw_wave, dtype=np.float64)
    if w.ndim != 1 or len(w) != SR or not np.isfinite(w).all():
        raise ValueError('Expected exactly one finite mono 1 s window; no padding.')
    w = w - w.mean()
    raw_rms = float(np.sqrt(np.mean(w * w)))
    if raw_rms < 1e-8:
        return np.zeros(SR, dtype=np.float64), 0.0
    y = signal.sosfiltfilt(_SOS_HP, w)
    y = signal.sosfiltfilt(_SOS_LP, y)
    for b_n, a_n in _NOTCHES:
        y = signal.filtfilt(b_n, a_n, y)
    y = y - y.mean()
    cond_rms = float(np.sqrt(np.mean(y * y)))
    if cond_rms < 1e-8:
        return np.zeros(SR, dtype=np.float64), 0.0
    return y / cond_rms, cond_rms


def stft_power(wave, nperseg, hop):
    f, t, z = signal.stft(wave, fs=SR, nperseg=nperseg, noverlap=nperseg - hop,
                          boundary=None, padded=False)
    return f, t, np.abs(z) ** 2


def local_noise_floor(power, frequency_spacing):
    width = int(round(NOISE_MEDIAN_HZ / frequency_spacing))
    if width % 2 == 0:
        width += 1
    size = width if np.ndim(power) == 1 else (width, 1)
    floor = median_filter(power, size=size, mode='nearest')
    snr_db = 10.0 * np.log10(np.maximum(power, 1e-12) / np.maximum(floor, 1e-12))
    return floor, snr_db


def centred_cepstrum(short_power, short_floor):
    relative = short_power / np.maximum(short_floor, 1e-12)
    mel_db = 10.0 * np.log10(np.maximum(MEL_BANK_SHORT @ relative, 1e-6))
    mel_db -= np.median(mel_db, axis=0, keepdims=True)
    m_axis = np.linspace(-1.0, 1.0, mel_db.shape[0])[:, None]
    mel_db = mel_db - m_axis * (np.sum(m_axis * mel_db, axis=0, keepdims=True) / np.sum(m_axis ** 2))
    cep = dct(mel_db, type=2, axis=0, norm='ortho')[:13]
    means = cep.mean(1)
    means[0] = 0.0
    cep_scale = max(float(np.sqrt(np.mean(means[1:] ** 2))), 1.5)
    means[1:] = means[1:] / cep_scale * 2.5
    delta = np.diff(cep, axis=1)
    return means, cep.std(1), np.abs(delta).mean(1), delta.std(1)


def harmonic_comb(f, power, floor):
    sigma2 = np.maximum(floor / LN2, 1e-12)
    ratio = np.clip(power / sigma2, 0.0, 60.0)
    evidence = np.log1p(np.maximum(ratio - 1.50, 0.0))
    ev_smooth = maximum_filter1d(evidence, size=3, axis=0, mode='nearest')
    positions = F0_GRID[None, :] * np.arange(1, 5)[:, None]
    right = np.clip(np.searchsorted(f, positions), 1, len(f) - 1)
    fraction = (positions - f[right - 1]) / (f[right] - f[right - 1])
    components = (1.0 - fraction[:, :, None]) * ev_smooth[right - 1] + fraction[:, :, None] * ev_smooth[right]
    weights = np.array([1.0, 0.75, 0.45, 0.25])[:, None, None]
    subharm_penalty = 0.65 * np.maximum(components[1] - components[0], 0.0) + 0.45 * np.maximum(components[2] - components[0], 0.0)
    shs = np.maximum(np.sum(components * weights, axis=0) - subharm_penalty, 0.0)
    return shs


def harmonic_energy_ratios(f, denoised_power, f0):
    energies = np.array([denoised_power[np.abs(f - f0 * h) <= 18.0].sum() for h in (1, 2, 3)])
    return energies[1] / max(energies[0], 1e-12), energies[2] / max(energies[0], 1e-12)


def _streaming_subband_descriptors(w):
    frames = w[:15872].reshape(31, 512)
    frames = frames - frames.mean(axis=-1, keepdims=True)
    power = np.abs(rfft(frames * _HANN512, axis=-1)) ** 2
    selected = power[:, 8:100]  # 250..3125 Hz
    total = np.maximum(selected.sum(axis=-1), 1e-20)
    bands = selected.reshape(31, 23, 4).sum(axis=-1) / total[:, None]
    log_bands = np.log10(np.maximum(bands, 1e-8))
    std_bands = log_bands.std(axis=0)
    low_band = power[:, 9:25]  # 281.25..781.25 Hz
    peak_ratio = np.max(low_band, axis=-1) / total
    peak_freq = (low_band.argmax(axis=-1) + 9) * (SR / 512.0 / 1000.0)
    return [
        float(peak_ratio.mean()),
        float(peak_ratio.std()),
        float(peak_freq.mean()),
        float(peak_freq.std()),
        float(std_bands[:5].mean()),
        float(std_bands[5:13].mean()),
        float(std_bands[13:].mean()),
    ]


def window_features(raw_wave):
    w, cond_rms = precondition_window(raw_wave)
    f, _, long = stft_power(w, LONG_NPERSEG, LONG_HOP)
    fs, ts, short = stft_power(w, SHORT_NPERSEG, SHORT_HOP)
    long_floor, _ = local_noise_floor(long, 5.0)
    short_floor, short_snr = local_noise_floor(short, 10.0)
    mean_power = long.mean(1)
    mean_floor, snr_db = local_noise_floor(mean_power, 5.0)
    denoised = np.maximum(mean_power - mean_floor, 0.0)

    psd = np.interp(PSD_GRID, f, np.clip(snr_db, -6.0, 45.0))
    psd -= np.median(psd)
    psd -= _LOG_AXIS * (float(np.dot(_LOG_AXIS, psd)) / _LOG_AXIS_NORM)
    psd -= np.median(psd)

    contrast = np.clip(short_snr, -6.0, 45.0)
    c_mean = np.interp(CONTRAST_GRID, fs, contrast.mean(1))
    c_mean -= np.median(c_mean)
    c_std = np.interp(CONTRAST_GRID, fs, contrast.std(1))

    cep = centred_cepstrum(short, short_floor)
    long_shs = harmonic_comb(f, long, long_floor)
    short_shs = harmonic_comb(fs, short, short_floor)
    profile = long_shs.mean(1)
    peaks, _ = signal.find_peaks(profile, distance=5)
    if not len(peaks):
        peaks = np.array([int(profile.argmax())])
    ranked = sorted(peaks, key=lambda i: (-profile[i], int(i)))[:3]
    peak_features = []
    for i in ranked:
        r2, r3 = harmonic_energy_ratios(f, denoised, F0_GRID[i])
        peak_features.extend([F0_GRID[i], profile[i], min(r2, 100.0), min(r3, 100.0)])
    peak_features.extend([0.0] * (12 - len(peak_features)))

    short_track = F0_GRID[short_shs.argmax(0)]
    long_track = F0_GRID[long_shs.argmax(0)]
    q = profile / max(float(profile.sum()), 1e-12)
    band = (f >= 250.0) & (f <= 3150.0)
    spectral = mean_power[band] / max(float(mean_power[band].sum()), 1e-12)
    centroid = float(np.dot(f[band], spectral))
    bandwidth = float(np.sqrt(np.dot((f[band] - centroid) ** 2, spectral)))

    mask = np.zeros(len(f), bool)
    for i in ranked:
        for h in (1, 2, 3, 4):
            mask |= np.abs(f - F0_GRID[i] * h) <= 18.0
    energy_fraction = float(denoised[mask & band].sum() / max(float(denoised[band].sum()), 1e-12))
    salience = float(np.clip(0.1 + profile.max() / 3.0 * np.sqrt(energy_fraction), 0.1, 3.0))
    if cond_rms < 1e-8:
        salience = 0.1

    mid = denoised[(f >= 450.0) & (f <= 1600.0)].sum()
    low = denoised[(f >= 250.0) & (f < 450.0)].sum()
    whole = denoised[band].sum()
    long_times = np.arange(len(long_track)) * LONG_HOP / SR + LONG_NPERSEG / (2.0 * SR)
    aligned = np.interp(ts, long_times, long_track)

    summary = [
        np.median(short_track), np.std(short_track), *np.quantile(short_track, [0.1, 0.9]),
        -np.sum(q * np.log(np.maximum(q, 1e-12))) / np.log(len(q)), energy_fraction,
        np.sqrt(np.dot((F0_GRID - np.dot(F0_GRID, q)) ** 2, q)),
        profile[ranked[1]] / max(profile[ranked[0]], 0.1) if len(ranked) > 1 else 0.0,
        np.exp(np.log(np.maximum(spectral, 1e-12)).mean()) / max(spectral.mean(), 1e-12),
        centroid, bandwidth, salience, mid / max(float(whole), 1e-12),
        np.log1p(mid / max(float(low), 1e-12)), np.median(long_track), np.std(long_track),
        np.mean(np.abs(short_track - aligned)),
        *_streaming_subband_descriptors(w),
    ]
    features = np.r_[psd, c_mean, c_std, *cep, peak_features, summary].astype(np.float32)
    if len(features) != len(FEATURE_NAMES) or not np.isfinite(features).all():
        raise ValueError('Nonfinite or misaligned features.')
    return features, salience


def extract_features(windows):
    rows = [window_features(w) for w in windows]
    return np.stack([r[0] for r in rows]), np.array([r[1] for r in rows])
