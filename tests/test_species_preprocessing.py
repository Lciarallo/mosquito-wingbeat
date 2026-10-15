"""Signal behavior and compatibility contracts for the preprocessing comparison."""
import numpy as np
import pytest
import torch

from species_preprocessing import SpeciesFrontend, VARIANTS, pcen
from torres_lstm import TorresLSTM


@pytest.mark.parametrize("variant", VARIANTS)
def test_frontends_are_finite_for_silence_and_preserve_time_and_gain(variant):
    rng = torch.Generator().manual_seed(4)
    wave = torch.randn(2, 15872, generator=rng)
    frontend = SpeciesFrontend(variant)
    assert frontend(wave).shape == (2, 55, 64)
    assert torch.isfinite(frontend(torch.zeros_like(wave))).all()
    torch.testing.assert_close(frontend(wave), frontend(2 * wave + .3), rtol=1e-4, atol=2e-4)


@pytest.mark.parametrize("variant", VARIANTS)
def test_processing_one_window_does_not_use_other_recordings(variant):
    wave = torch.randn(2, 15872, generator=torch.Generator().manual_seed(9))
    frontend = SpeciesFrontend(variant)
    torch.testing.assert_close(frontend(wave[:1]), frontend(wave)[:1], rtol=1e-5, atol=1e-5)


def test_pcen_matches_scalar_recurrence_and_has_no_future_frame_dependency():
    energy = torch.tensor([[[1., 4., 2., 3.]]])
    result = pcen(energy, .25)
    state = 1.
    expected = []
    for index, value in enumerate(energy.flatten().tolist()):
        if index:
            state = .75 * state + .25 * value
        expected.append((value / (1e-6 + state) ** .98 + 2) ** .5 - 2 ** .5)
    np.testing.assert_allclose(result.flatten(), expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(result[..., :2], pcen(energy[..., :2], .25))


def test_recurrent_checkpoint_roundtrip_supports_64_bands_and_legacy_32():
    for features in (32, 64):
        model = TorresLSTM(hidden=8, features=features, classes=20, bidirectional=True).eval()
        restored = TorresLSTM(**model.config).eval()
        restored.load_state_dict(model.state_dict())
        x = torch.randn(2, 55, features)
        with torch.inference_mode():
            torch.testing.assert_close(model(x), restored(x))
        assert model(x).shape == (2, 20)


def test_logmel_peak_tracks_frequency_without_pitch_translation():
    frontend = SpeciesFrontend()
    times = torch.arange(15872) / 16000
    waves = torch.stack([torch.sin(2 * torch.pi * frequency * times) for frequency in (300, 600)])
    peak = frontend(waves).mean(1).argmax(1)
    assert peak[1] > peak[0]
    assert frontend.config["fft_bin_spacing_hz"] == 7.8125


def test_invalid_audio_cannot_be_silently_padded_or_repeated():
    with pytest.raises(ValueError, match="15872"):
        SpeciesFrontend()(torch.zeros(2, 8000))
