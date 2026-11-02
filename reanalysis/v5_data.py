"""v5 (exploratória): caches de forma de onda a 8 kHz.

  optical   Wingbeats óptico (BioDCASE D5), 6 espécies, ~265 mil clipes de 5.000 amostras -> int16.
  acoustic  Linhas de mosquito do desenvolvimento da v4 (o teste reservado da v3 sai em load_development)
            + Funchal (figshare). Forma de onda decodificada, 8 kHz, passa-faixa 100-3.400 Hz, arquivo inteiro.

Nenhum rótulo entra no pré-processamento. Saída em data/cache/v5/ (fora do Git).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import sosfiltfilt

from reanalysis import frontend as fe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/cache/v5"
D5 = ROOT / "data/external/biodcase_dev/Development_data/raw_audio"
FUNCHAL = ROOT / "data/external/additional/figshare/Mosquitos Audio Samples"
# Ordem das espécies no BioDCASE 2026 (split_summary.json): S1..S9
BDC_SPECIES = {1: "Aedes aegypti", 2: "Aedes albopictus", 3: "Culex quinquefasciatus", 4: "Anopheles gambiae",
               5: "Anopheles arabiensis", 6: "Anopheles dirus", 7: "Culex pipiens", 8: "Anopheles minimus",
               9: "Anopheles stephensi"}
MAX_SECONDS = None  # arquivo inteiro, como no cache do Perch da v3


def optical():
    files = sorted(D5.glob("S_*_D_5_*.wav"))
    species = np.array([int(p.name.split("_")[1]) for p in files], dtype=np.int8)
    waves = np.zeros((len(files), 5000), dtype=np.int16)

    def read(i):
        x, rate = sf.read(str(files[i]), dtype="int16")
        if rate != 8000 or x.ndim != 1:
            raise ValueError(f"unexpected format {files[i]}")
        waves[i, :min(5000, len(x))] = x[:5000]

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(read, range(len(files))))
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "optical_waves.npy", waves)
    pd.DataFrame(dict(file=[p.name for p in files], species_id=species,
                      species=[BDC_SPECIES[s] for s in species])).to_csv(OUT / "optical_meta.csv", index=False)
    print(pd.Series(species).map(BDC_SPECIES).value_counts().to_dict())


def funchal_device(folder):
    """Aparelho do Funchal pela taxa de amostragem da pasta. '48khz' testado antes de '8khz', que está contido nele."""
    return "48k" if "48khz" in folder else "44k" if "44.1khz" in folder else "8k"


def funchal_frame():
    rows = []
    for p in sorted(FUNCHAL.rglob("*.wav")):
        rate = funchal_device(p.parent.name)
        aedes = p.parent.parent.name == "Aedes Aegypti"
        rows.append(dict(path=str(p.relative_to(ROOT)), origin="funchal", target=2 if aedes else 1,
                         species="Aedes aegypti" if aedes else p.parent.parent.name,
                         cluster_merged=f"funchal|{rate}", sex=np.nan))
    return pd.DataFrame(rows)


def acoustic():
    from reanalysis import v4_train as v4
    frame, _ = v4.load_development()
    dev = frame.loc[frame.target > 0, ["path", "origin", "target", "species", "cluster_merged"]].copy()
    dev["sex"] = frame.loc[frame.target > 0].path.str.extract(r"_\d+([FM])_")[0]
    meta = pd.concat([dev, funchal_frame()], ignore_index=True)
    meta["cluster_merged"] = meta.cluster_merged.astype(str)

    def load(path):
        x = fe.decode(ROOT / path)
        if len(x) > 30:
            x = sosfiltfilt(fe.SOS, x)
        return np.clip(np.nan_to_num(x), -1, 1).astype(np.float16)

    with ThreadPoolExecutor(12) as pool:
        waves = list(pool.map(load, meta.path))
    lengths = np.array([len(w) for w in waves])
    offsets = np.r_[0, np.cumsum(lengths)[:-1]]
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "acoustic_waves.npy", np.concatenate(waves))
    meta["offset"], meta["length"] = offsets, lengths
    meta.to_csv(OUT / "acoustic_meta.csv", index=False)
    print(meta.groupby(["origin", "target"]).agg(n=("path", "size"), clusters=("cluster_merged", "nunique")))


def acoustic_wave(waves, row):
    return waves[row.offset: row.offset + row.length].astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["optical", "acoustic"])
    {"optical": optical, "acoustic": acoustic}[parser.parse_args().stage]()
