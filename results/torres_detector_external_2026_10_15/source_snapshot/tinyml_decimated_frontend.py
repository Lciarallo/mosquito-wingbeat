"""Decimated Zoom-FFT Frontend for TinyML on Arduino Nano 33 BLE Sense.

Standard Arduino firmware operates at 16 kHz with 512-point FFT:
Bin resolution = 16000 / 512 = 31.25 Hz.
At this resolution, distinguishing Aedes aegypti (550-610 Hz) from Anopheles (450-520 Hz)
is coarse and prone to temperature and boundary quantization errors.

Decimated Zoom-FFT approach:
1. Low-pass anti-aliasing IIR filter (order 4, cutoff 1800 Hz).
2. Decimation 4:1 (subsamples from 16 kHz to 4 kHz).
3. 512-point FFT on the 4 kHz signal:
   Bin resolution = 4000 / 512 = 7.8125 Hz (4x finer resolution!).
   Frequency range = 0 to 2000 Hz (perfectly covering F0, 2xF0, 3xF0 for all mosquito species).
4. RAM requirement: Exactly 512 float/int16 words (identical to current firmware, <4 KB RAM).
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, resample_poly
from scipy.fft import rfft, rfftfreq

SAMPLE_RATE_IN = 16000
DECIMATION_FACTOR = 4
SAMPLE_RATE_DECIMATED = SAMPLE_RATE_IN // DECIMATION_FACTOR  # 4000 Hz
FFT_SIZE = 512
BIN_RESOLUTION = SAMPLE_RATE_DECIMATED / FFT_SIZE  # 7.8125 Hz
NUM_BINS = FFT_SIZE // 2 + 1  # 257 bins (0 to 2000 Hz)

# 4th order Butterworth anti-aliasing filter at 1800 Hz cutoff (fs=16000)
# Expressed as 2 cascaded biquads (Second-Order Sections) for embedded precision
SOS_ANTIALIAS = butter(4, 1800.0, btype='lowpass', fs=SAMPLE_RATE_IN, output='sos')

# Fixed Hann window for 512 points
HANN_512 = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(FFT_SIZE) / (FFT_SIZE - 1))).astype(np.float32)

def decimate_audio(wave_16k: np.ndarray) -> np.ndarray:
    """Filter and downsample from 16 kHz to 4 kHz."""
    filtered = sosfilt(SOS_ANTIALIAS, wave_16k)
    return filtered[::DECIMATION_FACTOR]

def compute_decimated_spectrum(decimated_wave_4k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute high-resolution 512-point FFT on 4 kHz signal (7.8125 Hz/bin)."""
    if len(decimated_wave_4k) < FFT_SIZE:
        raise ValueError(f"Need at least {FFT_SIZE} samples of 4 kHz audio.")
    frame = decimated_wave_4k[:FFT_SIZE].astype(np.float32)
    frame = frame - np.mean(frame)
    spectrum = rfft(frame * HANN_512, n=FFT_SIZE)
    power = np.abs(spectrum)**2
    freqs = rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE_DECIMATED)
    return power, freqs

def extract_decimated_features(wave_16k: np.ndarray) -> dict[str, float]:
    """Extract fine-grained bioacoustic features from 16 kHz audio via decimation."""
    wave_4k = decimate_audio(wave_16k)
    power, freqs = compute_decimated_spectrum(wave_4k)
    
    # 1. Total power in wingbeat range (200 - 1800 Hz)
    valid_mask = (freqs >= 200.0) & (freqs <= 1800.0)
    total_power = np.maximum(np.sum(power[valid_mask]), 1e-12)
    norm_power = power / total_power
    
    # 2. Precise fundamental peak search in 300 - 800 Hz (bins ~38 to ~102)
    f0_mask = (freqs >= 300.0) & (freqs <= 800.0)
    f0_idx = np.argmax(norm_power[f0_mask])
    f0_hz = float(freqs[f0_mask][f0_idx])
    f0_power = float(norm_power[f0_mask][f0_idx])
    
    # 3. Narrow sub-bands with 31.25 Hz width (matching 4 decimated bins each!)
    # Band from 300 to 900 Hz in 31.25 Hz slices (19 fine bands)
    fine_bands = []
    for low in np.arange(300.0, 900.0, 31.25):
        high = low + 31.25
        b_mask = (freqs >= low) & (freqs < high)
        fine_bands.append(float(np.sum(norm_power[b_mask])))
        
    # 4. Harmonic 2xF0 tracking with 7.8 Hz resolution
    h2_target = 2.0 * f0_hz
    h2_mask = (freqs >= h2_target - 16.0) & (freqs <= h2_target + 16.0)
    h2_power = float(np.sum(norm_power[h2_mask])) if np.any(h2_mask) else 1e-6
    h2_ratio = h2_power / max(f0_power, 1e-6)
    
    # 5. Spectral sharpness (peak to median ratio in wingbeat band)
    median_power = float(np.median(norm_power[valid_mask]))
    sharpness = f0_power / max(median_power, 1e-6)
    
    return {
        'decimated_f0_hz': f0_hz,
        'decimated_f0_power': f0_power,
        'decimated_h2_ratio': h2_ratio,
        'decimated_sharpness': sharpness,
        'fine_bands': fine_bands
    }

def generate_cpp_decimator_header() -> str:
    """Generate C++ biquad IIR filter coefficients for Arduino Nano 33 BLE Sense."""
    b0_1, b1_1, b2_1 = SOS_ANTIALIAS[0, :3]
    a0_1, a1_1, a2_1 = SOS_ANTIALIAS[0, 3:]
    b0_2, b1_2, b2_2 = SOS_ANTIALIAS[1, :3]
    a0_2, a1_2, a2_2 = SOS_ANTIALIAS[1, 3:]
    
    cpp = f"""// Decimated Zoom-FFT Frontend for Arduino Nano 33 BLE Sense
// Decimates 16 kHz PDM audio to 4 kHz with 4th-order Butterworth anti-aliasing IIR
// 512-point FFT at 4 kHz provides 7.8125 Hz bin resolution.
// Memory: exactly 512 float words (2 KB RAM).

#ifndef DECIMATED_FRONTEND_H
#define DECIMATED_FRONTEND_H

#include <stdint.h>
#include <math.h>

#define DECIMATION_RATIO 4
#define DECIMATED_FFT_SIZE 512
#define DECIMATED_BIN_HZ 7.8125f

// Biquad section 1 coefficients
static const float BIQUAD1_B[3] = {{{b0_1:.8e}f, {b1_1:.8e}f, {b2_1:.8e}f}};
static const float BIQUAD1_A[2] = {{{a1_1:.8e}f, {a2_1:.8e}f}};

// Biquad section 2 coefficients
static const float BIQUAD2_B[3] = {{{b0_2:.8e}f, {b1_2:.8e}f, {b2_2:.8e}f}};
static const float BIQUAD2_A[2] = {{{a1_2:.8e}f, {a2_2:.8e}f}};

struct DecimatorState {{
    float s1_w1, s1_w2;
    float s2_w1, s2_w2;
    uint8_t sample_counter;
}};

inline void init_decimator(DecimatorState* st) {{
    st->s1_w1 = st->s1_w2 = 0.0f;
    st->s2_w1 = st->s2_w2 = 0.0f;
    st->sample_counter = 0;
}}

// Process single 16 kHz sample. Returns true when a 4 kHz sample is ready.
inline bool process_decimator_sample(DecimatorState* st, float in_sample, float* out_sample) {{
    // Section 1 (Direct Form II Transposed)
    float w1 = in_sample - BIQUAD1_A[0] * st->s1_w1 - BIQUAD1_A[1] * st->s1_w2;
    float y1 = BIQUAD1_B[0] * w1 + BIQUAD1_B[1] * st->s1_w1 + BIQUAD1_B[2] * st->s1_w2;
    st->s1_w2 = st->s1_w1;
    st->s1_w1 = w1;

    // Section 2
    float w2 = y1 - BIQUAD2_A[0] * st->s2_w1 - BIQUAD2_A[1] * st->s2_w2;
    float y2 = BIQUAD2_B[0] * w2 + BIQUAD2_B[1] * st->s2_w1 + BIQUAD2_B[2] * st->s2_w2;
    st->s2_w2 = st->s2_w1;
    st->s2_w1 = w2;

    st->sample_counter++;
    if (st->sample_counter >= DECIMATION_RATIO) {{
        st->sample_counter = 0;
        *out_sample = y2;
        return true;
    }}
    return false;
}}

#endif // DECIMATED_FRONTEND_H
"""
    return cpp
