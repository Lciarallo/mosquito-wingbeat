"""Streaming-compatible, gain-invariant FFT frontend for the Arduino detector.

16 kHz mono, non-overlapping 512-sample Hann frames, 31 frames per decision
(0.992 s). No recording-level normalization or future samples are required.
The matching C++ implementation keeps only one FFT frame and running moments.
"""
from __future__ import annotations
import numpy as np
from scipy.fft import rfft

SAMPLE_RATE, FRAME_SIZE, WINDOW_FRAMES, BANDS = 16000, 512, 31, 31
WINDOW_SAMPLES = FRAME_SIZE * WINDOW_FRAMES
VERSION = "streaming-fft512-ratios-1"
FEATURE_NAMES = ([f"band_log_mean_{125*(i+1)}_{125*(i+2)}" for i in range(BANDS)] +
                 [f"band_log_std_{125*(i+1)}_{125*(i+2)}" for i in range(BANDS)] +
                 ["flatness_mean", "flatness_std", "low_peak_ratio_mean", "low_peak_ratio_std",
                  "low_peak_frequency_mean", "low_peak_frequency_std"])
HANN = (0.5 - 0.5 * np.cos(2*np.pi*np.arange(FRAME_SIZE)/(FRAME_SIZE-1))).astype("float32")

def pcm16(waves):
    """One fixed gain per clip for replay; real PDM PCM is consumed directly.

    Corpus cache is RMS-normalized float audio, often outside [-1,1]. Scaling
    to 0.9 full scale prevents artificial clipping in the PCM equivalence test.
    Ratios are scale-invariant except at the explicit numerical floor.
    """
    waves = np.asarray(waves, dtype="float32")
    maximum = np.max(np.abs(waves), axis=-1, keepdims=True)
    return np.rint(waves / np.maximum(maximum, 1e-12) * (0.9*32767)).astype("int16")

def features(waves, replay_pcm=True):
    waves = np.atleast_2d(np.asarray(waves))
    if waves.shape[1] < WINDOW_SAMPLES:
        raise ValueError(f"São necessárias {WINDOW_SAMPLES} amostras contíguas.")
    if replay_pcm and waves.dtype != np.int16:
        waves = pcm16(waves)
    frames = waves[:, :WINDOW_SAMPLES].astype("float32").reshape(-1, WINDOW_FRAMES, FRAME_SIZE)
    frames -= frames.mean(axis=-1, keepdims=True)
    transformed = rfft(frames*HANN, axis=-1)
    power = np.abs(transformed)**2
    # FFT bins 4..127 correspond to 125..3968.75 Hz. Four bins per band.
    selected = power[:, :, 4:128]
    total = np.maximum(selected.sum(axis=-1), 1e-20)
    energy = selected.reshape(-1, WINDOW_FRAMES, BANDS, 4).sum(axis=-1)
    ratios = energy / total[..., None]
    log_ratios = np.log10(np.maximum(ratios, 1e-8))
    relative_bins = selected / total[..., None]
    flatness = np.exp(np.log(np.maximum(relative_bins, 1e-12)).mean(axis=-1)) * selected.shape[-1]
    low_band = power[:, :, 7:29]  # 218.75..875 Hz; bins fixed in C++.
    peak = np.max(low_band, axis=-1) / total
    frequency = (low_band.argmax(axis=-1)+7) * (SAMPLE_RATE/FRAME_SIZE/1000.)
    return np.c_[log_ratios.mean(axis=1), log_ratios.std(axis=1), flatness.mean(axis=1),
                 flatness.std(axis=1), peak.mean(axis=1), peak.std(axis=1),
                 frequency.mean(axis=1), frequency.std(axis=1)].astype("float32")

def persistence(scores, paths, starts, threshold, required=2, history=3):
    """Causal vote; reset across paths and gaps between sampled corpus windows.

    Requires `history` contiguous observed windows. The corpus sampled at most
    60 windows/file, frequently with gaps; those are not continuous recordings.
    """
    scores, starts = np.asarray(scores), np.asarray(starts)
    decision = np.zeros(len(scores), dtype=bool)
    eligible = np.zeros(len(scores), dtype=bool)
    state, previous_path, previous_start = [], None, None
    for i, (score, path, start) in enumerate(zip(scores, paths, starts)):
        if path != previous_path or previous_start is None or abs(start-previous_start-1.) > 1e-5:
            state = []
        state.append(bool(score >= threshold))
        state = state[-history:]
        eligible[i] = len(state) == history
        decision[i] = eligible[i] and sum(state) >= required
        previous_path, previous_start = path, start
    return decision, eligible
