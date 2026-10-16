"""Bioacoustic feature extraction with temperature invariance and swarm dynamics.

Features:
1. Adaptive F0 tracking via Harmonic Product Spectrum (HPS) and subharmonic summation.
2. Temperature-invariant harmonic ratios: H2/H1, H3/H1, H4/H1, H5/H1, harmonic decay slope, and HNR.
3. Swarm intermodulation dynamics: Hilbert envelope demodulation of wingbeat band (400-750 Hz),
   modulation spectrum (2-40 Hz beating), envelope dispersion and peak cluster width.
4. Channel/microphone normalization via Cepstral Mean Subtraction (CMS).
"""
from __future__ import annotations
import numpy as np
from scipy.signal import hilbert, butter, sosfilt
from scipy.fft import rfft, rfftfreq

SR = 16000
CARRIER_LOW, CARRIER_HIGH = 400.0, 750.0  # Wingbeat fundamental band for Culicidae
SOS_CARRIER = butter(4, [CARRIER_LOW, CARRIER_HIGH], btype='bandpass', fs=SR, output='sos')

BIOACOUSTIC_FEATURE_NAMES = [
    # Temperature-invariant harmonic descriptors
    'hps_f0_hz',
    'harmonic_h2_ratio',
    'harmonic_h3_ratio',
    'harmonic_h4_ratio',
    'harmonic_h5_ratio',
    'harmonic_decay_slope',
    'harmonic_to_noise_ratio_db',
    'in_band_spectral_flatness',
    'out_band_spectral_flatness',
    'band_concentration_ratio',
    # Swarm intermodulation & envelope modulation dynamics (2-40 Hz)
    'envelope_mod_energy_ratio',
    'envelope_mod_peak_hz',
    'envelope_mod_entropy',
    'envelope_crest_factor',
    'swarm_dispersion_index',
    'carrier_energy_ratio'
]

def compute_hps_f0(power_spectrum: np.ndarray, freqs: np.ndarray,
                   min_f0: float = 250.0, max_f0: float = 900.0,
                   num_harmonics: int = 4) -> tuple[float, float]:
    """Estimate fundamental frequency F0 using Harmonic Product Spectrum (HPS).
    
    HPS downsamples the power spectrum by integer factors and multiplies.
    The fundamental harmonic frequency reinforces while non-harmonic noise cancels out.
    """
    n_bins = len(power_spectrum)
    hps = power_spectrum.copy()
    
    for h in range(2, num_harmonics + 1):
        # Decimate spectrum by factor h
        decimated = power_spectrum[::h]
        hps[:len(decimated)] *= decimated
        hps[len(decimated):] = 0.0

    valid_mask = (freqs >= min_f0) & (freqs <= max_f0)
    if not np.any(valid_mask):
        return 0.0, 0.0
    
    masked_hps = hps[valid_mask]
    masked_freqs = freqs[valid_mask]
    
    peak_idx = np.argmax(masked_hps)
    best_f0 = float(masked_freqs[peak_idx])
    hps_strength = float(masked_hps[peak_idx])
    return best_f0, hps_strength

def extract_temperature_invariant_harmonics(power: np.ndarray, freqs: np.ndarray,
                                            f0: float, bin_width: float) -> dict[str, float]:
    """Calculate relative energy of harmonics H2, H3, H4, H5 relative to H1.
    
    Because wingbeat mechanics govern the harmonic overtone decay regardless of 
    whether ambient temperature shifts F0 from 480 Hz to 600 Hz, these ratios
    are temperature-invariant physical descriptors.
    """
    audible = (freqs >= 200.0) & (freqs <= 3500.0)
    total_audible = max(float(np.sum(power[audible])), 1e-12)
    if f0 <= 0.0:
        ratios = [0.0, 0.0, 0.0, 0.0]
        slope = 0.0
        hnr_db = 0.0
    else:
        tolerance = max(25.0, bin_width * 2)
        harmonic_powers = []

        for k in range(1, 6):
            target_f = k * f0
            if target_f >= freqs[-1]:
                harmonic_powers.append(1e-12)
                continue
            h_mask = np.abs(freqs - target_f) <= tolerance
            if np.any(h_mask):
                harmonic_powers.append(float(np.sum(power[h_mask])))
            else:
                harmonic_powers.append(1e-12)

        h1 = max(harmonic_powers[0], 1e-12)
        ratios = [harmonic_powers[k] / h1 for k in range(1, 5)]

        # Estimate harmonic decay slope (linear fit of log harmonic energy vs harmonic index)
        indices = np.arange(1, 6)
        log_h = np.log10(np.clip(harmonic_powers, 1e-12, None))
        # slope of log10 energy per harmonic step
        slope = float(np.polyfit(indices, log_h, 1)[0])

        # Harmonic-to-Noise Ratio (HNR)
        # Sum of all harmonic powers vs remaining power in the 200-3500 Hz range
        harmonics_sum = sum(harmonic_powers)
        noise_energy = max(total_audible - harmonics_sum, 1e-12)
        hnr_db = float(10.0 * np.log10(harmonics_sum / noise_energy))
    
    # Carrier band vs outside carrier band
    carrier_mask = (freqs >= CARRIER_LOW) & (freqs <= CARRIER_HIGH)
    out_mask = (freqs >= 100.0) & (freqs <= 4000.0) & (~carrier_mask)
    
    p_carrier = power[carrier_mask] + 1e-12
    p_out = power[out_mask] + 1e-12
    
    flatness_carrier = float(np.exp(np.mean(np.log(p_carrier))) / np.mean(p_carrier))
    flatness_out = float(np.exp(np.mean(np.log(p_out))) / np.mean(p_out))
    concentration = float(np.sum(p_carrier) / total_audible)
    
    return {
        'hps_f0_hz': f0,
        'harmonic_h2_ratio': ratios[0],
        'harmonic_h3_ratio': ratios[1],
        'harmonic_h4_ratio': ratios[2],
        'harmonic_h5_ratio': ratios[3],
        'harmonic_decay_slope': slope,
        'harmonic_to_noise_ratio_db': hnr_db,
        'in_band_spectral_flatness': flatness_carrier,
        'out_band_spectral_flatness': flatness_out,
        'band_concentration_ratio': concentration
    }

def extract_swarm_modulation(wave: np.ndarray, sr: int = SR) -> dict[str, float]:
    """Extract envelope modulation features to distinguish swarms from single mosquitoes.
    
    In a swarm, multiple mosquitoes beating wings at slightly differing frequencies
    (e.g., 590 Hz, 600 Hz, 610 Hz) cause interferometric amplitude modulation (beating)
    at difference frequencies in the 2-40 Hz range.
    """
    # 1. Bandpass filter in the mosquito wingbeat carrier band
    filtered = sosfilt(SOS_CARRIER, wave)
    
    # Carrier energy ratio
    total_energy = float(np.sum(wave**2) + 1e-12)
    carrier_energy = float(np.sum(filtered**2))
    carrier_ratio = carrier_energy / total_energy
    
    # 2. Instantaneous envelope via Hilbert transform
    analytic = hilbert(filtered)
    envelope = np.abs(analytic)
    mean_env = float(np.mean(envelope) + 1e-12)
    std_env = float(np.std(envelope))
    crest_factor = float(np.max(envelope) / mean_env)
    
    # Remove DC for modulation spectrum
    norm_env = (envelope - mean_env) / (std_env + 1e-12)
    
    # 3. Modulation spectrum (FFT of envelope)
    n_env = len(norm_env)
    env_fft = np.abs(rfft(norm_env))
    env_freqs = rfftfreq(n_env, 1.0 / sr)
    
    # Look at modulation frequencies 2 to 40 Hz (biological flight beat notes)
    mod_mask = (env_freqs >= 2.0) & (env_freqs <= 40.0)
    all_mask = (env_freqs >= 0.5) & (env_freqs <= 100.0)
    
    total_mod_power = max(float(np.sum(env_fft[all_mask]**2)), 1e-12)
    band_mod_power = float(np.sum(env_fft[mod_mask]**2))
    mod_energy_ratio = band_mod_power / total_mod_power
    
    if np.any(mod_mask):
        peak_mod_idx = np.argmax(env_fft[mod_mask])
        peak_mod_hz = float(env_freqs[mod_mask][peak_mod_idx])
        # Entropy of modulation spectrum in 2-40 Hz
        p_mod = env_fft[mod_mask] / (np.sum(env_fft[mod_mask]) + 1e-12)
        p_mod = p_mod[p_mod > 1e-12]
        mod_entropy = 0.0 if len(p_mod) <= 1 else float(-np.sum(p_mod * np.log(p_mod)) / np.log(len(p_mod)))
    else:
        peak_mod_hz = 0.0
        mod_entropy = 1.0

    # Swarm dispersion index: ratio of standard deviation to mean of envelope
    # High dispersion + high modulation energy in 2-40 Hz indicates multi-insect beating
    dispersion_index = std_env / mean_env
    
    return {
        'envelope_mod_energy_ratio': mod_energy_ratio,
        'envelope_mod_peak_hz': peak_mod_hz,
        'envelope_mod_entropy': mod_entropy,
        'envelope_crest_factor': crest_factor,
        'swarm_dispersion_index': dispersion_index,
        'carrier_energy_ratio': carrier_ratio
    }

def apply_cepstral_mean_subtraction(log_energies: np.ndarray) -> np.ndarray:
    """Apply Cepstral Mean Subtraction (CMS) across a batch or temporal sequence.
    
    Removes stationary linear filter effects (channel H(w) of microphone/enclosure)
    by subtracting the mean log-spectral energy across windows.
    """
    if log_energies.ndim == 1:
        return log_energies - np.mean(log_energies)
    return log_energies - np.mean(log_energies, axis=0, keepdims=True)

def extract_bioacoustic_features_for_wave(wave: np.ndarray, sr: int = SR,
                                         n_fft: int = 2048) -> np.ndarray:
    """Extract full bioacoustic vector (16 features) for a single window of audio."""
    # Ensure zero-mean
    wave = wave.astype(np.float32)
    wave = wave - np.mean(wave)
    rms = np.sqrt(np.mean(wave**2))
    if rms < 1e-6:
        return np.zeros(len(BIOACOUSTIC_FEATURE_NAMES), dtype=np.float32)
    
    # 1. Power spectrum
    window = np.hanning(len(wave)).astype(np.float32)
    spectrum = rfft(wave * window, n=n_fft)
    freqs = rfftfreq(n_fft, 1.0 / sr)
    power = np.abs(spectrum)**2
    bin_width = freqs[1] - freqs[0]
    
    # 2. HPS F0 estimation
    f0, _ = compute_hps_f0(power, freqs, min_f0=250.0, max_f0=900.0)
    
    # 3. Temperature-invariant harmonic features
    harmonics_dict = extract_temperature_invariant_harmonics(power, freqs, f0, bin_width)
    
    # 4. Swarm dynamics features
    swarm_dict = extract_swarm_modulation(wave, sr=sr)
    
    # Combine in exact order
    merged = {**harmonics_dict, **swarm_dict}
    vector = np.array([merged[k] for k in BIOACOUSTIC_FEATURE_NAMES], dtype=np.float32)
    return vector

def extract_batch_bioacoustic_features(waves: np.ndarray, sr: int = SR) -> np.ndarray:
    """Extract bioacoustic features for a 2D batch of windows (N x L)."""
    waves = np.atleast_2d(waves)
    results = [extract_bioacoustic_features_for_wave(w, sr=sr) for w in waves]
    return np.stack(results, axis=0)
