"""Reanalysis v2 - step 1: file-level manifest with explicit independence clusters.

One row per audio file. The unit of independence is the `cluster`, the
recording environment / session, not the file:

* Dryad (Abuzz) mosquitoes: lab+device directory (e.g. CDC_T209, GoogleNexus).
* Dryad "noise" (Pete_dataset): original noise recording before slicing.
* HumBug: country|place|device|location_type|recording day (metadata CSV).
* ESC-50: original freesound clip id (takes of one clip stay together).
* MosquitoSong: individual (the LowMic file is a 2nd microphone of the same flight).
* UFRGS: folder (female / male / not Aedes).

Excluded with reason: Torres 2025 (no raw audio, 100% Aedes), Figshare (not
an acoustic microphone corpus), unknown species, private Drive swarm audio.
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SEGMENTS = ROOT / "data/metadata/ternary_segments.csv.gz"
HUMBUG_META = ROOT / "data/external/humbug/neurips_2021_zenodo_0_0_1.csv"
OUT = ROOT / "results/v2"
CLASSES = ["no_mosquito", "other_mosquito", "aedes_aegypti"]
EXTERNAL_SOURCES = {"mosquitosong", "ufrgs"}


def probe(path):
    try:
        info = sf.info(str(path))
        return info.samplerate, info.duration
    except Exception:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                              "stream=sample_rate:format=duration", "-of", "json", str(path)],
                             capture_output=True, text=True, check=True).stdout
        j = json.loads(out)
        return int(j["streams"][0]["sample_rate"]), float(j["format"]["duration"])


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def build():
    seg = pd.read_csv(SEGMENTS, low_memory=False)
    files = seg.drop_duplicates("path")[["path", "label", "domain", "group", "session", "id"]].copy()
    files["label"] = files.label.astype(str)
    reasons = pd.Series("", index=files.index)
    reasons[files.domain.eq("figshare")] = "figshare_not_acoustic_microphone"
    reasons[files.label.str.lower().isin(["unknown_mosquito", "unknown", "mosquito"])] = "unknown_species"
    reasons[files.path.str.contains("drive_audio|drive_test|private", case=False)] = "private_drive_audio"
    excluded = files[reasons.ne("")].assign(reason=reasons[reasons.ne("")])
    files = files[reasons.eq("")].copy()

    is_noise = files.label.str.lower().eq("noise") | files.domain.eq("esc50")
    is_aedes = files.label.eq("Aedes aegypti") | files.path.str.contains("/Aedes_aegypti_(?:female|male)/", regex=True)
    files["target"] = np.where(is_noise, 0, np.where(is_aedes, 2, 1)).astype(int)
    files["class_name"] = np.asarray(CLASSES)[files.target]
    files.loc[files.domain.eq("dryad") & is_noise, "domain"] = "pete_noise"
    files = files.rename(columns={"domain": "source", "label": "species"})

    hm = pd.read_csv(HUMBUG_META)
    stem = pd.to_numeric(files.path.str.extract(r"/([^/]+)\.wav$")[0], errors="coerce")
    given = pd.to_numeric(files.id, errors="coerce")
    fid = given.where(given.isin(hm.id), stem)
    files["humbug_id"] = np.where(files.source.eq("humbug"), fid, np.nan)
    meta = hm.set_index("id")
    cluster, device, setting = [], [], []
    for row in files.itertuples():
        if row.source == "dryad":
            env = str(row.session).split("/")[-1]
            cluster.append("dryad|" + env); device.append(env.split("_")[-1]); setting.append(env)
        elif row.source == "pete_noise":
            cluster.append("pete|" + str(row.group).split("|")[-1]); device.append("unknown"); setting.append("pete")
        elif row.source == "esc50":
            clip = re.search(r"/\d-(\d+)-[A-Z]-\d+\.wav$", row.path).group(1)
            cluster.append("esc50|" + clip); device.append("freesound"); setting.append("esc50")
        elif row.source == "humbug":
            if pd.notna(row.humbug_id) and row.humbug_id in meta.index:
                m = meta.loc[row.humbug_id]
                env = "|".join(str(m[c]) for c in ["country", "place", "device_type", "location_type"])
                day = str(m.record_datetime)[:8]
                cluster.append("humbug|" + env + "|" + day); device.append(str(m.device_type)); setting.append(env)
            else:
                cluster.append("humbug|nometa|" + str(row.group)); device.append("unknown"); setting.append("nometa")
        elif row.source == "mosquitosong":
            name = Path(row.path).name
            ind = re.sub(r"_cut\d+.*$", "", name, flags=re.IGNORECASE).replace("_LowMic", "").replace(" ", "")
            cluster.append("mosquitosong|" + ind); device.append("LowMic" if "LowMic" in name else "mic"); setting.append("mosquitosong")
        elif row.source == "ufrgs":
            cluster.append("ufrgs|" + Path(row.path).parent.name); device.append("phone"); setting.append("ufrgs")
        else:
            raise ValueError("Unknown source " + row.source)
    files["cluster"], files["device"], files["setting"] = cluster, device, setting
    files["role"] = np.where(files.source.isin(EXTERNAL_SOURCES), "external", "development")
    sr, dur = zip(*[probe(ROOT / p) for p in files.path])
    files["native_sr"], files["duration_s"] = sr, dur
    files["file_sha256"] = [sha256(ROOT / p) for p in files.path]
    files = files.drop(columns=["group", "session", "id"]).reset_index(drop=True)
    dup = files.duplicated("file_sha256", keep=False)
    for _, g in files[dup].groupby("file_sha256"):  # identical bytes must share a cluster
        if g.target.nunique() > 1:
            raise ValueError("Identical audio with conflicting labels: " + str(g.path.tolist()))
        files.loc[files.cluster.isin(g.cluster), "cluster"] = g.cluster.iloc[0]
    OUT.mkdir(parents=True, exist_ok=True)
    files.to_csv(OUT / "manifest_files.csv", index=False)
    excluded.to_csv(OUT / "excluded_files.csv", index=False)
    summary = files.groupby(["role", "source", "class_name"]).agg(
        files=("path", "size"), clusters=("cluster", "nunique"), hours=("duration_s", lambda x: x.sum() / 3600))
    summary.to_csv(OUT / "inventory_summary.csv")
    return files, excluded, summary


if __name__ == "__main__":
    f, e, s = build()
    print(s.round(3).to_string())
    print("excluded:", e.reason.value_counts().to_dict())
    print("duplicate-byte files:", int(f.duplicated("file_sha256", keep=False).sum()))
