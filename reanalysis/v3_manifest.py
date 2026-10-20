"""Detector v3 - manifest with duplicate components, corpus of origin, clusters and the locked test."""
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3"
SPECIES = {1: "Aedes aegypti", 2: "Aedes albopictus", 3: "Culex quinquefasciatus", 4: "Anopheles gambiae",
           5: "Anopheles arabiensis", 6: "Anopheles dirus", 7: "Culex pipiens", 8: "Anopheles minimus",
           9: "Anopheles stephensi"}
TRAIN_CORPORA = ["abuzz", "humbug", "mosquitosong"]


def build():
    lm, src = pickle.load(open(OUT / "landmarks_stage2.pkl", "rb"))
    links = pd.read_csv(OUT / "dup_links_stage2.csv")
    v2 = pd.read_csv(ROOT / "results/v2/manifest_files.csv").set_index("path")
    hm = pd.read_csv(ROOT / "data/external/humbug/neurips_2021_zenodo_0_0_1.csv").set_index("id")
    rows = []
    for p in lm:
        s = src[p]
        r = dict(path=p, raw_source=s, pcm=lm[p][0])
        if p in v2.index:
            v = v2.loc[p]
            r.update(species=v.species, target=int(v.target), cluster=v.cluster, device=v.device,
                     corpus={"dryad": "abuzz"}.get(v.source, v.source), native_sr=v.native_sr)
        elif s.startswith("biodcase"):
            m = re.search(r"S_(\d+)_D_(\d+)_(\d+)", p); S, D = int(m[1]), int(m[2])
            r.update(species=SPECIES[S], target=2 if S == 1 else 1, cluster=f"bdc|D{D}|S{S}", device="unknown",
                     corpus="bdc_unknown", bdc_domain=D)
        else:  # humbugdb_full
            hid = int(Path(p).stem)
            if hid not in hm.index:
                r.update(exclude="humbug_id_without_metadata"); rows.append(r); continue
            mt = hm.loc[hid]
            env = "|".join(str(mt[c]) for c in ["country", "place", "device_type", "location_type"])
            r.update(humbug_id=hid, cluster="humbug|" + env + "|" + str(mt.record_datetime)[:8], device=str(mt.device_type),
                     corpus="humbug", setting=env)
            if mt.sound_type == "background":
                r.update(species="noise", target=0)
            elif mt.sound_type == "mosquito" and isinstance(mt.species, str):
                r.update(species=mt.species, target=2 if "aegypti" in mt.species else 1)
            else:
                r.update(exclude="humbug_audio_or_unknown_species")
        rows.append(r)
    f = pd.DataFrame(rows)
    # --- components (exact PCM + landmark links)
    par = {p: p for p in f.path}
    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    def uni(a, b):
        a, b = find(a), find(b)
        if a != b: par[b] = a
    for _, g in f.groupby("pcm"):
        for q in g.path.iloc[1:]: uni(g.path.iloc[0], q)
    for a, b in zip(links.path, links.other): uni(a, b)
    f["component"] = f.path.map(find)
    # --- label conflicts inside a component
    lab = f[f.exclude.isna()] if "exclude" in f else f
    conflict = lab.groupby("component").target.nunique()
    f.loc[f.component.isin(conflict[conflict > 1].index) & f.exclude.isna(), "exclude"] = "label_conflict_in_duplicate_component"
    # --- corpus of origin of the component
    def origin(g):
        c = set(g.corpus.dropna())
        for k in TRAIN_CORPORA:
            if k in c: return k
        if "ufrgs" in c or (g.get("bdc_domain") == 3).any(): return "locked_test_ufrgs"
        for k in ["esc50", "pete_noise", "bdc_unknown"]:
            if k in c: return k
        return "other"
    comp_origin = f.groupby("component").apply(origin)
    f["origin"] = f.component.map(comp_origin)
    # --- clusters: merge all clusters that share a component
    cpar = {c: c for c in f.cluster.dropna().unique()}
    def cfind(a):
        while cpar[a] != a:
            cpar[a] = cpar[cpar[a]]; a = cpar[a]
        return a
    for _, g in f.dropna(subset=["cluster"]).groupby("component"):
        cs = g.cluster.unique()
        known = [c for c in cs if not c.startswith("bdc|")]
        if known:  # placeholder BioDCASE clusters never bridge real recording sessions (amendment A1)
            cs = np.array(known)
            for p in g.path[g.cluster.str.startswith("bdc|")]:
                f.loc[f.path.eq(p), "cluster"] = known[0]
        for c in cs[1:]:
            a, b = cfind(cs[0]), cfind(c)
            if a != b: cpar[b] = a
    f["cluster_merged"] = f.cluster.map(lambda c: cfind(c) if isinstance(c, str) else None)
    # --- keep one representative file per component (the one with most audio), others marked duplicate
    f["role"] = np.where(f.origin.eq("locked_test_ufrgs"), "locked_test",
                np.where(f.origin.isin(["esc50", "pete_noise"]), "noise_train", "train"))
    f.loc[f.raw_source.eq("biodcase_D5"), "exclude"] = "optical_wingbeats_D5"
    f["n_hashes"] = f.path.map(lambda p: len(lm[p][1]))
    ok = f.exclude.isna()
    rep = f[ok].sort_values("n_hashes", ascending=False).groupby("component").path.first()
    f["representative"] = f.path.isin(rep.values)
    f.loc[ok & ~f.representative, "exclude"] = "duplicate_of_component_representative"
    keep = f[f.exclude.isna()].copy()
    durs = []
    for p in keep.path:
        try: durs.append(sf.info(str(ROOT / p)).duration)
        except Exception: durs.append(np.nan)
    keep["duration_s"] = durs
    keep = keep.drop(columns=["pcm"]).reset_index(drop=True)
    keep.to_csv(OUT / "manifest_v3.csv", index=False)
    f.drop(columns=["pcm"]).to_csv(OUT / "manifest_v3_all_with_exclusions.csv", index=False)
    return keep, f


if __name__ == "__main__":
    k, f = build()
    print(f.exclude.fillna("kept").value_counts().to_string())
    print(k.groupby(["role", "origin", "target"]).agg(files=("path", "size"), clusters=("cluster_merged", "nunique"),
                                                      hours=("duration_s", lambda x: x.sum() / 3600)).round(2).to_string())
