"""The private swarm must not enter the individual-flight detector corpus."""
from pathlib import Path

import pandas as pd
import pytest

from ternary_protocol import EXCLUDED_DRIVE_SHA256, excluded_drive_audio_mask, load_dataset


def test_private_swarm_aliases_and_renamed_byte_identity_are_excluded():
    frame = pd.DataFrame([
        {"domain": "Drive", "path": "copy.wav"},
        {"group": "drive:recording", "path": "copy.wav"},
        {"path": r"C:\data\drive_test\copy.wav"},
        {"path": "tmp/private-audio-test/download.audio"},
        {"path": "drive_audio.wav"},
        {"path": "renamed.wav", "content_sha256": EXCLUDED_DRIVE_SHA256},
        {"domain": "mosquitosong", "path": "public.wav", "download_url": "https://drive.google.com/public"},
        {"domain": "dryad", "path": "data/raw/Aedes/flight.wav"},
    ], index=range(10, 18))
    assert excluded_drive_audio_mask(frame).tolist() == [True] * 6 + [False, False]


def test_neural_reference_rejects_swarm_before_feature_loading(tmp_path):
    from train_ternary_lstm import load_inputs
    pd.DataFrame([{"domain": "Drive", "path": "drive_audio.wav"}]).to_csv(
        tmp_path / "segments.csv.gz", index=False)
    with pytest.raises(ValueError, match="Drive swarm audio is excluded"):
        load_inputs(tmp_path, "cpu")


def test_current_corpus_is_unchanged_and_contains_no_private_swarm():
    root = Path(__file__).resolve().parents[1]
    reference = pd.read_csv(root / "results/ternary_verified_2026_10_14_corrections/segments.csv.gz", low_memory=False)
    neural = pd.read_csv(root / "results/ternary_lstm_2026_10_15/segments.csv.gz", low_memory=False)
    current, _, _, stats = load_dataset()
    assert stats["excluded_drive_swarm"] == 0
    for frame in (reference, neural, current):
        assert len(frame) == 42029
        assert frame.group.nunique() == 1494
        assert set(frame.domain) == {"dryad", "humbug", "esc50"}
        assert not excluded_drive_audio_mask(frame).any()
        assert set(frame.row_id) == set(reference.row_id)


def test_source_transfer_rejects_private_swarm_before_reading_a_run(tmp_path):
    from evaluate_ternary_source_transfer import evaluate_source
    for source in ("Drive", "drive_test", "private_swarm"):
        with pytest.raises(ValueError, match="outside the scope"):
            evaluate_source(tmp_path / "nonexistent", source)
