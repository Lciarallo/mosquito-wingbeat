"""Frequency-preserving frontends used identically in fitting and inference."""
import math

import torch
from torch.nn import functional as F

from torres_lstm import MelSequence


VARIANTS = ("hires_logmel", "hires_contrast", "hires_pcen")


def pcen(energy, smoothing, alpha=.98, delta=2., exponent=.5, epsilon=1e-6):
    """Causal per-channel energy normalization, initialized from the first frame.

    Input shape is (..., frequency, time). All parameters are fixed, not fitted
    from validation/test inputs. Each call handles an independent audio window.
    """
    if energy.ndim < 2 or not energy.shape[-1] or not 0 < smoothing <= 1:
        raise ValueError("PCEN needs a nonempty time axis and smoothing in (0,1].")
    state = energy[..., 0]
    smooth = [state]
    for index in range(1, energy.shape[-1]):
        state = (1 - smoothing) * state + smoothing * energy[..., index]
        smooth.append(state)
    average = torch.stack(smooth, -1)
    return (energy / (epsilon + average).pow(alpha) + delta).pow(exponent) - delta ** exponent


class SpeciesFrontend(MelSequence):
    def __init__(self, variant="hires_logmel"):
        if variant not in VARIANTS:
            raise ValueError("Unknown species preprocessing: " + variant)
        super().__init__(n_fft=2048, bands=64)
        self.variant = variant
        self.config.update(variant=variant, version=1, frame_samples=15872,
                           fft_bin_spacing_hz=16000 / 2048, frame_stride_s=256 / 16000)
        if variant == "hires_contrast":
            self.config.update(contrast_bands=9, envelope_compensation=.5,
                               scaling="peak-relative log-mel minus 0.5 times local frequency envelope")
        elif variant == "hires_pcen":
            self.config.update(pcen_alpha=.98, pcen_delta=2., pcen_exponent=.5,
                               pcen_epsilon=1e-6, pcen_time_constant_s=.4,
                               pcen_smoothing=1 - math.exp(-256 / (16000 * .4)),
                               pcen_initialization="first frame of each independent window",
                               scaling="PCEN on RMS-normalized power mel, FFT divided by Hann sum")

    def mel_power(self, wave):
        if wave.ndim != 2 or wave.shape[-1] != 15872 or not torch.isfinite(wave).all():
            raise ValueError("Expected finite audio windows of 15872 samples at 16 kHz.")
        centered = wave.float() - wave.float().mean(-1, keepdim=True)
        normalized = centered / centered.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        stft = torch.stft(normalized, 2048, 256, window=self.window, center=False,
                          return_complex=True) / self.window.sum()
        return self.bank @ stft.abs().square()

    def forward(self, wave):
        energy = self.mel_power(wave)
        if self.variant == "hires_pcen":
            value = pcen(energy, self.config["pcen_smoothing"])
        else:
            db = 10 * torch.log10(energy.clamp_min(1e-12))
            value = (db - db.amax((-2, -1), keepdim=True)).clamp(-80, 0)
            if self.variant == "hires_contrast":
                envelope = F.avg_pool2d(value[:, None], (9, 1), stride=1,
                                        padding=(4, 0), count_include_pad=False)[:, 0]
                value = value - .5 * envelope
        return value.transpose(-1, -2).contiguous()
