"""Descritores acústicos usados pelo detector ternário em janelas de áudio."""

import numpy as np
from scipy.fft import rfft, dct

SR = 16000
FFT = 2048
HOP = 256
BANDS = [f"band_{125 * i}_{125 * (i + 1)}" for i in range(1, 32)]
MEASURE_NAMES = BANDS + [f"cepstrum_{i}" for i in range(1, 20)] + [
    "centroid_hz", "bandwidth_hz", "rolloff85_hz", "flatness", "entropy",
    "peak_hz", "low_peak_hz", "harmonic2_ratio", "harmonic3_ratio",
    "low_band_ratio", "spectral_flux",
]
FEATURE_NAMES = [
    f"{measure}_{stat}"
    for stat in ["mean", "std"]
    for measure in MEASURE_NAMES
]


def features(waves, batch=128):
    """Extrai 122 descritores por janela; não replica nem preenche áudio curto."""
    waves = np.atleast_2d(waves)
    size = waves.shape[1]
    if size < 4800:
        raise ValueError("Need at least 0.3 s of real audio; no padding or repetition.")
    starts = np.arange(0, size - FFT + 1, HOP)
    freq = np.fft.rfftfreq(FFT, 1 / SR)
    valid = (freq >= 125) & (freq < 4000)
    fv = freq[valid]
    window = np.hanning(FFT).astype("float32")
    result = []
    for start in range(0, len(waves), batch):
        x = np.array(waves[start:start + batch], dtype="float32", copy=True)
        x -= x.mean(axis=1, keepdims=True)
        x /= np.sqrt(np.mean(x * x, axis=1, keepdims=True)).clip(1e-6)
        frames = np.stack([x[:, i:i + FFT] for i in starts], axis=1)
        power = np.abs(rfft(frames * window, axis=-1)) ** 2
        p = power[..., valid]
        p = p / p.sum(axis=-1, keepdims=True).clip(1e-20)
        energy = np.stack([
            p[..., (fv >= 125 * i) & (fv < 125 * (i + 1))].sum(axis=-1)
            for i in range(1, 32)
        ], axis=-1)
        log = np.log10(energy.clip(1e-8))
        cep = dct(log, type=2, norm="ortho", axis=-1)[..., 1:20]
        centroid = (p * fv).sum(-1)
        bandwidth = np.sqrt((p * (fv - centroid[..., None]) ** 2).sum(-1))
        rolloff = fv[np.argmax(np.cumsum(p, axis=-1) >= .85, axis=-1)]
        flatness = np.exp(np.log(p.clip(1e-12)).mean(-1)) * len(fv)
        entropy = -(p * np.log(p.clip(1e-12))).sum(-1) / np.log(len(fv))
        peak = fv[p.argmax(-1)]
        low = (fv >= 200) & (fv <= 900)
        lowp = p[..., low]
        lowfreq = fv[low][lowp.argmax(-1)]
        peakamp = lowp.max(-1).clip(1e-8)
        h2 = (p * (np.abs(fv[None, None, :] - 2 * lowfreq[..., None]) <= 16)).sum(-1) / peakamp
        h3 = (p * (np.abs(fv[None, None, :] - 3 * lowfreq[..., None]) <= 16)).sum(-1) / peakamp
        flux = np.concatenate([
            np.zeros((*p.shape[:1], 1)),
            np.sqrt(((p[:, 1:] - p[:, :-1]) ** 2).mean(-1)),
        ], axis=1)
        stat = np.stack([
            centroid, bandwidth, rolloff, flatness, entropy, peak, lowfreq,
            h2, h3, lowp.sum(-1), flux,
        ], axis=-1)
        all_features = np.concatenate([log, cep, stat], axis=-1)
        result.append(np.concatenate([
            all_features.mean(1), all_features.std(1),
        ], axis=1).astype("float32"))
    answer = np.concatenate(result)
    if answer.shape[1] != len(FEATURE_NAMES) or not np.isfinite(answer).all():
        raise ValueError("Invalid acoustic feature matrix.")
    return answer
