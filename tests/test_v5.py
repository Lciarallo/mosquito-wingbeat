"""v5 exploratória: isolamento (teste reservado da v3, UFRGS e óptico fora da avaliação), agrupamento do Funchal,
pesos por fonte e formatos da CNN. Os testes de dados locais são pulados quando os caches não existem."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from reanalysis import v5_data, v5_eval, v5_optical

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data/cache/v5/acoustic_meta.csv"
MANIFEST = ROOT / "results/v3_continuation/manifest.csv"


def test_funchal_device_48k_is_not_8k():
    assert v5_data.funchal_device("48khz") == "48k"
    assert v5_data.funchal_device("Mix Pipiens & Quinquefasciatus_48khz") == "48k"
    assert v5_data.funchal_device("Quinquefasciatus_44.1khz") == "44k"
    assert v5_data.funchal_device("8khz") == "8k"


def test_source_weights_balance_class_and_source():
    origin = np.array(["a"] * 10 + ["b"] * 2 + ["a"] * 3 + ["c"] * 7, dtype=object)
    y = np.array([1] * 12 + [0] * 10)
    w = v5_eval.source_weights(origin, y)
    for c in [0, 1]:
        per_source = [w[(y == c) & (origin == s)].sum() for s in np.unique(origin[y == c])]
        assert np.allclose(per_source, per_source[0])
    assert np.isclose(w[y == 1].sum(), w[y == 0].sum())


def test_cnn_shapes_keep_frequency_axis():
    x = torch.randn(3, v5_optical.WIN)
    s = v5_optical.spectrogram(x)
    assert s.shape == (3, 1, 107, 79)
    net = v5_optical.Net(6).eval()
    with torch.no_grad():
        assert net(s).shape == (3, 6)
        assert net.features(s).shape == (3, 128)


@pytest.mark.skipif(not (META.exists() and MANIFEST.exists()), reason="cache local da v5 ausente")
def test_evaluation_rows_exclude_reserved_ufrgs_and_optical():
    meta = pd.read_csv(META)
    man = pd.read_csv(MANIFEST)
    reserved = set(man.loc[man.role.isin(["locked_test", "locked_noise", "quarantine_holdout_related"]), "path"])
    assert not set(meta.path) & reserved
    assert not meta.path.str.contains("ufrgs", case=False).any()
    # D5 é o óptico (só treino); S_1_D_3 e S_2_D_3 são o teste reservado da v3 (Aedes do insetário da UFRGS).
    assert not meta.path.str.contains("_D_5_|S_1_D_3_|S_2_D_3_").any()
    assert set(meta.origin) == {"abuzz", "humbug", "mosquitosong", "bdc_unknown", "funchal"}
    assert meta.loc[meta.origin.eq("funchal"), "cluster_merged"].nunique() == 3
