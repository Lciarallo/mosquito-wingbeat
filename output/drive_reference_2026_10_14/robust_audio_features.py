"""Extração de Características Bioacústicas Robustas, Detecção de Enxames e Invariância a Canal.

Revisão Matemática (v2 — Detecção Multi-Pitch de Enxames de Aedes):
1. Pré-condicionamento Físico de Banda:
   - Filtro Passa-Altas Butterworth de 4ª ordem em 180 Hz: elimina o rumble sub-grave
     (< 150 Hz) que concentrava 94,7% da energia bruta em gravações reais e esmagava
     o denominador da Razão de Energia Harmônica (HER).
   - Pente de Filtros Notch IIR (Q = 30) nos harmônicos da rede elétrica (60, 120,
     180, 240, 300 e 360 Hz): suprime tons estacionários de equipamentos/rede (como o
     6º harmônico em 359,4 Hz) que antes induziam salto bimodal no argmax frame a frame.
2. Subtração de Piso Espectral com Correção de Viés Qui-Quadrado (Rayleigh/Exponencial)
   e Limiar de Significância de Bonferroni:
   - Em quadros STFT de ruído gaussiano complexo, P(f,t) ~ Exp(1/sigma^2), cuja mediana
     teórica é sigma^2 * ln(2). Assim, sigma^2_hat = median_filter(P) / ln(2).
   - Aplica limiar de significância corrigido para múltiplos testes (Bonferroni sobre a
     grade de 80 candidatos de F0) combinando piso espectral local (vizinhança de 195 Hz)
     e piso mediano global da janela, eliminando falsos positivos em trechos silenciosos.
3. Decomposição Harmônica Multi-Pitch (Subharmonic Summation - SHS) com Acoplamento 2F0:
   - Avalia simultaneamente até K=3 fundamentais F0 em [380, 780] Hz e seus harmônicos
     2F0 e 3F0, permitindo detectar enxames (coros com múltiplos mosquitos voando juntos,
     onde a dispersão IQR/std de F0 ultrapassa 85 Hz) sem rejeitá-los como ruído.
4. Razão de Sinal-Ruído Normalizada por Largura de Banda (Aedes Bandwidth-Normalized SNR):
   - Compara a densidade espectral média por bin nas bandas harmônicas de Aedes
     (F0: 420–740 Hz; 2F0: 840–1480 Hz) contra a banda de fundo inferior (200–390 Hz),
     garantindo expectativa 0 dB sob ruído branco plano.
"""
from __future__ import annotations

import numpy as np
import scipy.signal as signal
from scipy.fftpack import dct
from scipy.ndimage import median_filter
from scipy.special import expit

import arduino_frontend

SAMPLE_RATE = 16000
FRAME_SIZE = 512
WINDOW_FRAMES = 31
WINDOW_SAMPLES = FRAME_SIZE * WINDOW_FRAMES  # 15872 amostras (0,992 s @ 16 kHz)
LN2 = float(np.log(2.0))


def compute_cmvn(mfcc_frames: np.ndarray) -> np.ndarray:
    """Aplica normalização de média e variância cepstral ao longo dos frames de tempo."""
    mean = np.mean(mfcc_frames, axis=1, keepdims=True)
    std = np.std(mfcc_frames, axis=1, keepdims=True)
    return (mfcc_frames - mean) / np.maximum(std, 1e-6)


def precondition_bioacoustic_wave(wave: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Remove rumble sub-180 Hz e harmônicos estacionários de 60 Hz (60..360 Hz)."""
    w = np.asarray(wave, dtype=np.float64).ravel()
    scale = 32768.0 if np.max(np.abs(w)) > 2.0 else 1.0
    w_norm = (w / scale) - np.mean(w / scale)
    if np.sqrt(np.mean(w_norm ** 2)) < 1e-5:
        return np.zeros_like(w, dtype=np.float32)

    sos_hp = signal.butter(4, 180.0 / (sr / 2.0), btype="highpass", output="sos")
    y = signal.sosfiltfilt(sos_hp, w_norm)
    for h_freq in (60.0, 120.0, 180.0, 240.0, 300.0, 360.0):
        b_n, a_n = signal.iirnotch(h_freq, Q=30.0, fs=sr)
        y = signal.filtfilt(b_n, a_n, y)

    return (y * scale).astype(np.float32)


def analyze_harmonic_flutter(wave: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """Avalia presença de voo isolado ou enxame de Aedes vs ruído estacionário/fundo."""
    w_raw = np.asarray(wave, dtype=np.float64).ravel()
    if np.max(np.abs(w_raw)) > 2.0:
        w_norm = w_raw / 32768.0
    else:
        w_norm = w_raw.copy()
    w_norm = w_norm - np.mean(w_norm)
    raw_rms = float(np.sqrt(np.mean(w_norm ** 2)))
    clip_frac = float(np.mean(np.abs(w_norm) >= (32700.0 / 32768.0)))

    if raw_rms < 5e-4:
        return {
            "f0_mean": 0.0,
            "f0_std": 0.0,
            "f0_iqr": 0.0,
            "second_f0_hz": 0.0,
            "is_electrical_60hz": False,
            "is_biological_flutter": False,
            "is_aedes_swarm": False,
            "is_clipped": False,
            "harmonic_ratio": 0.0,
            "h2_h1_ratio": 0.0,
            "aedes_snr_db": -20.0,
            "comb_strength": 0.0,
            "active_duty_cycle": 0.0,
            "swarm_peak_count": 0,
            "swarm_score": 0.0,
            "regime": "SILENCIO_DIGITAL",
        }

    sos_hp = signal.butter(4, 180.0 / (sr / 2.0), btype="highpass", output="sos")
    y = signal.sosfiltfilt(sos_hp, w_norm)

    f_w, psd_pre = signal.welch(y, fs=sr, nperseg=2048, noverlap=1536)
    hum_bins = (
        (np.abs(f_w - 240.0) <= 10.0)
        | (np.abs(f_w - 300.0) <= 10.0)
        | (np.abs(f_w - 360.0) <= 10.0)
    )
    bio_bins = (f_w >= 400.0) & (f_w <= 1500.0)
    hum_vs_bio_db = float(
        10.0
        * np.log10(
            max(float(np.mean(psd_pre[hum_bins])), 1e-18)
            / max(float(np.mean(psd_pre[bio_bins])), 1e-18)
        )
    )

    for h_freq in (60.0, 120.0, 180.0, 240.0, 300.0, 360.0):
        b_n, a_n = signal.iirnotch(h_freq, Q=30.0, fs=sr)
        y = signal.filtfilt(b_n, a_n, y)

    f, t, Zxx = signal.stft(y, fs=sr, nperseg=2048, noverlap=1536, boundary=None, padded=False)
    P = np.abs(Zxx) ** 2
    band_all = (f >= 200.0) & (f <= 3500.0)
    global_med_power = float(np.median(P[band_all, :])) / LN2

    local_med = median_filter(P, size=(25, 1), mode="nearest") / LN2
    sigma2_eff = np.maximum(local_med, 0.35 * global_med_power + 1e-18)

    snr_bonf = np.maximum(P / sigma2_eff - 5.5, 0.0)

    P_mean = np.mean(P, axis=1)
    floor_welch = np.maximum(
        median_filter(P_mean, size=25, mode="nearest"),
        0.35 * float(np.median(P_mean[band_all])) + 1e-18,
    )
    welch_snr = np.maximum(P_mean / floor_welch - 2.0, 0.0)
    welch_excess = np.maximum(P_mean - 1.5 * floor_welch, 0.0)

    f0_grid = np.arange(250.0, 855.0, 5.0)
    comb_tf = np.zeros((len(f0_grid), P.shape[1]), dtype=np.float64)
    comb_welch = np.zeros(len(f0_grid), dtype=np.float64)

    for k, f0 in enumerate(f0_grid):
        idx1 = int(np.argmin(np.abs(f - f0)))
        idx2 = int(np.argmin(np.abs(f - 2.0 * f0)))
        idx3 = int(np.argmin(np.abs(f - 3.0 * f0)))

        s1 = np.max(snr_bonf[max(0, idx1 - 1) : min(len(f), idx1 + 2), :], axis=0)
        s2 = np.max(snr_bonf[max(0, idx2 - 2) : min(len(f), idx2 + 3), :], axis=0)
        s3 = np.max(snr_bonf[max(0, idx3 - 2) : min(len(f), idx3 + 3), :], axis=0)
        comb_tf[k, :] = np.log1p(s1) + 0.80 * np.log1p(s2) + 0.40 * np.log1p(s3)

        w1 = np.max(welch_snr[max(0, idx1 - 1) : min(len(f), idx1 + 2)])
        w2 = np.max(welch_snr[max(0, idx2 - 2) : min(len(f), idx2 + 3)])
        w3 = np.max(welch_snr[max(0, idx3 - 2) : min(len(f), idx3 + 3)])
        coupling = 1.35 if (w1 > 0.30 and w2 > 0.15) else 0.80
        comb_welch[k] = coupling * (np.log1p(w1) + 0.85 * np.log1p(w2) + 0.40 * np.log1p(w3))

    bio_mask = (f0_grid >= 380.0) & (f0_grid <= 780.0)
    bio_grid = f0_grid[bio_mask]
    bio_welch = comb_welch[bio_mask]
    bio_tf = comb_tf[bio_mask, :]

    peaks, _ = signal.find_peaks(bio_welch, height=0.25, prominence=0.10, distance=4)
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(bio_welch))])
    ranked = peaks[np.argsort(-bio_welch[peaks])][:3]

    f0_primary = float(bio_grid[ranked[0]])
    second_f0 = (
        float(bio_grid[ranked[1]])
        if len(ranked) > 1 and bio_welch[ranked[1]] >= 0.25
        else 0.0
    )
    comb_strength = float(bio_welch[ranked[0]])
    swarm_peak_count = int(np.sum(bio_welch[peaks] >= 0.38))

    frame_max = np.max(bio_tf, axis=0)
    frame_f0 = bio_grid[np.argmax(bio_tf, axis=0)]
    active_frames = frame_max >= 0.45
    active_duty = float(np.mean(active_frames))

    if np.sum(active_frames) >= 2:
        f0_mean = float(np.median(frame_f0[active_frames]))
        f0_std = float(np.std(frame_f0[active_frames]))
        f0_iqr = float(
            np.percentile(frame_f0[active_frames], 75)
            - np.percentile(frame_f0[active_frames], 25)
        )
    else:
        f0_mean = f0_primary
        f0_std = 0.0
        f0_iqr = 0.0

    mask_aedes_f0 = (f >= 420.0) & (f <= 740.0)
    mask_aedes_2f0 = (f >= 840.0) & (f <= 1480.0)
    mask_low_bg = (f >= 200.0) & (f < 390.0)

    mean_p_aedes = 0.65 * float(np.mean(P_mean[mask_aedes_f0])) + 0.35 * float(
        np.mean(P_mean[mask_aedes_2f0])
    )
    mean_p_low = float(np.mean(P_mean[mask_low_bg]))
    aedes_snr_db = float(10.0 * np.log10(max(mean_p_aedes, 1e-18) / max(mean_p_low, 1e-18)))

    total_welch_bio = float(np.sum(welch_excess[band_all])) + 1e-15
    harm_mask = np.zeros(len(f), dtype=bool)
    for pk_i in ranked:
        pf0 = float(bio_grid[pk_i])
        for h in (1, 2, 3):
            harm_mask |= np.abs(f - h * pf0) <= 25.0
    her_swarm = (
        float(np.clip(np.sum(welch_excess[harm_mask & band_all]) / total_welch_bio, 0.0, 1.0))
        if comb_strength > 0.10
        else 0.0
    )

    e_h1 = sum(
        float(np.sum(welch_excess[np.abs(f - float(bio_grid[i])) <= 25.0]))
        for i in ranked[:2]
    )
    e_h2 = sum(
        float(np.sum(welch_excess[np.abs(f - 2.0 * float(bio_grid[i])) <= 35.0]))
        for i in ranked[:2]
    )
    h2_h1_ratio = float(e_h2 / max(e_h1, 1e-15)) if e_h1 > 0 else 0.0

    is_clipped = bool(clip_frac > 0.01)
    is_60hz = bool(hum_vs_bio_db >= 2.5 and aedes_snr_db < 0.5 and comb_strength < 0.80)

    logit_score = (
        1.60 * (comb_strength - 0.80)
        + 0.55 * (aedes_snr_db - 0.70)
        + 2.60 * (active_duty - 0.28)
        + 1.40 * (her_swarm - 0.18)
        - 3.00 * float(is_60hz)
        - 3.50 * float(is_clipped)
    )
    swarm_score = float(expit(logit_score))

    comb_ok = bool(comb_strength >= 0.50 or (aedes_snr_db >= 3.0 and comb_strength >= 0.30))
    is_bio_flight = bool(
        not is_clipped
        and not is_60hz
        and swarm_score >= 0.50
        and comb_ok
        and aedes_snr_db >= -0.50
    )

    is_aedes_swarm = bool(
        is_bio_flight
        and 410.0 <= f0_mean <= 750.0
        and aedes_snr_db >= 1.00
        and (swarm_peak_count >= 2 or active_duty >= 0.45 or h2_h1_ratio >= 0.12 or aedes_snr_db >= 3.0)
    )

    if is_clipped:
        regime = "SATURACAO_CLIPPING"
    elif is_aedes_swarm:
        regime = "ENXAME_AEDES_ATIVO"
    elif is_bio_flight:
        regime = "VOO_ISOLADO_CURTO"
    elif is_60hz:
        regime = "RUIDO_ELETRICO_ESTACIONARIO"
    else:
        regime = "AUSENTE_FUNDO"

    return {
        "f0_mean": round(f0_mean, 1),
        "f0_std": round(f0_std, 1),
        "f0_iqr": round(f0_iqr, 1),
        "second_f0_hz": round(second_f0, 1),
        "is_electrical_60hz": is_60hz,
        "is_biological_flutter": is_bio_flight,
        "is_aedes_swarm": is_aedes_swarm,
        "is_clipped": is_clipped,
        "harmonic_ratio": round(her_swarm, 4),
        "h2_h1_ratio": round(h2_h1_ratio, 3),
        "aedes_snr_db": round(aedes_snr_db, 2),
        "comb_strength": round(comb_strength, 2),
        "active_duty_cycle": round(active_duty, 3),
        "swarm_peak_count": swarm_peak_count,
        "swarm_score": round(swarm_score, 4),
        "regime": regime,
    }


def extract_robust_features(waves: np.ndarray, apply_cmvn: bool = True) -> np.ndarray:
    """Extrai características robustas combinando streaming FFT pré-condicionado e métricas de enxame."""
    waves = np.atleast_2d(np.asarray(waves, dtype="float32"))
    clean_waves = np.stack([precondition_bioacoustic_wave(w, sr=SAMPLE_RATE) for w in waves])
    base_feats = arduino_frontend.features(clean_waves, replay_pcm=True)

    extra_feats = []
    for wave in waves:
        analysis = analyze_harmonic_flutter(wave, sr=SAMPLE_RATE)
        extra_feats.append(
            [
                analysis["f0_mean"] / 1000.0,
                analysis["f0_std"] / 100.0,
                float(analysis["is_electrical_60hz"]),
                float(analysis["is_biological_flutter"]),
                analysis["harmonic_ratio"],
            ]
        )

    extra_arr = np.array(extra_feats, dtype="float32")
    return np.c_[base_feats, extra_arr]
