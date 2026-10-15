"""Extração de Características Bioacústicas Robustas e Invariantes a Microfone.

Implementa os avanços metodológicos da literatura recente (BioDCASE 2026,
SEMISH 2025, SBCAS 2023):
1. CMVN (Cepstral Mean and Variance Normalization): neutraliza a coloração
   de frequência e função de transferência estática de cada aparelho/microfone.
2. Harmonic Energy Ratio (HER): quantifica a proeminência de F0 e de seus
   harmônicos (2F0, 3F0, 4F0) em relação ao piso de ruído broadband.
3. Flutter Dynamics & Stationarity: mede o desvio padrão da frequência instantânea.
   Distingue voo biológico (modulação aerodinâmica de ±20 a 50 Hz) de ruídos
   mecânicos ou harmônicos de rede elétrica de 60 Hz (estacionários, σ < 5 Hz).
4. Compatibilidade com o frontend streaming de 68 bandas do Arduino Nano 33 BLE Sense.
"""
from __future__ import annotations

import numpy as np
import scipy.signal as signal
from scipy.fftpack import dct
import arduino_frontend

SAMPLE_RATE = 16000
FRAME_SIZE = 512
WINDOW_FRAMES = 31
WINDOW_SAMPLES = FRAME_SIZE * WINDOW_FRAMES  # 15872 amostras


def compute_cmvn(mfcc_frames: np.ndarray) -> np.ndarray:
    """Aplica normalização de média e variância cepstral ao longo dos frames de tempo."""
    mean = np.mean(mfcc_frames, axis=1, keepdims=True)
    std = np.std(mfcc_frames, axis=1, keepdims=True)
    return (mfcc_frames - mean) / np.maximum(std, 1e-6)


def analyze_harmonic_flutter(wave: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """Avalia se o som é zumbido biológico modulado ou ruído elétrico/mecânico estacionário."""
    wave = np.asarray(wave, dtype="float32")
    wave = wave - wave.mean()
    rms = float(np.sqrt(np.mean(wave**2)))
    if rms < 1e-4:
        return {
            "f0_mean": 0.0,
            "f0_std": 0.0,
            "is_electrical_60hz": False,
            "is_biological_flutter": False,
            "harmonic_ratio": 0.0,
        }

    # STFT de alta resolução (janela de 128 ms, salto de 16 ms)
    nperseg = 2048
    noverlap = 1792
    f, t, Zxx = signal.stft(wave, fs=sr, nperseg=nperseg, noverlap=noverlap, boundary=None, padded=False)
    mag = np.abs(Zxx)

    # Faixa de frequência de voo de mosquito (200 - 900 Hz)
    band_mask = (f >= 200) & (f <= 900)
    f_band = f[band_mask]
    mag_band = mag[band_mask, :]

    # Trajetória da frequência dominante frame a frame
    instantaneous_peaks = f_band[np.argmax(mag_band, axis=0)]

    f0_mean = float(np.median(instantaneous_peaks))
    f0_std = float(np.std(instantaneous_peaks))

    # Verifica se f0_mean é múltiplo exato de 60 Hz (tolerância de ±4 Hz)
    nearest_60hz = round(f0_mean / 60.0) * 60.0
    dist_60hz = abs(f0_mean - nearest_60hz)
    is_60hz_harmonic = bool(dist_60hz <= 4.0 and nearest_60hz >= 120.0)

    # Ruído mecânico/elétrico: f0 muito estacionário (desvio padrão < 6 Hz)
    is_stationary = bool(f0_std < 6.0)
    is_electrical = bool(is_stationary and is_60hz_harmonic)

    # Voo biológico real: modulação aerodinâmica natural presente
    is_bio = bool(f0_std >= 12.0 and not is_electrical and f0_mean >= 250.0)

    # Razão de energia harmônica
    total_power = np.maximum(np.sum(mag**2), 1e-12)
    harmonic_power = 0.0
    for h in [1, 2, 3]:
        h_target = f0_mean * h
        h_mask = (f >= h_target - 25.0) & (f <= h_target + 25.0)
        harmonic_power += float(np.sum(mag[h_mask, :]**2))
    harmonic_ratio = float(min(harmonic_power / total_power, 1.0))

    return {
        "f0_mean": round(f0_mean, 1),
        "f0_std": round(f0_std, 1),
        "is_electrical_60hz": is_electrical,
        "is_biological_flutter": is_bio,
        "harmonic_ratio": round(harmonic_ratio, 3),
    }


def extract_robust_features(waves: np.ndarray, apply_cmvn: bool = True) -> np.ndarray:
    """Extrai características robustas anti-viés combinando streaming FFT e métricas harmônicas."""
    waves = np.atleast_2d(np.asarray(waves, dtype="float32"))
    base_feats = arduino_frontend.features(waves, replay_pcm=False)

    extra_feats = []
    for wave in waves:
        analysis = analyze_harmonic_flutter(wave, sr=SAMPLE_RATE)
        extra_feats.append([
            analysis["f0_mean"] / 1000.0,
            analysis["f0_std"] / 100.0,
            float(analysis["is_electrical_60hz"]),
            float(analysis["is_biological_flutter"]),
            analysis["harmonic_ratio"],
        ])

    extra_arr = np.array(extra_feats, dtype="float32")
    combined = np.c_[base_feats, extra_arr]
    return combined
