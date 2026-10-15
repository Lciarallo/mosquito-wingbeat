"""Held-out real noise probe of the compact presence detector (weak labels)."""
import json
import joblib
import numpy as np
import pandas as pd
import arduino_frontend as frontend
from train_arduino import ROOT,OUT,SEED,calibrated
from improve_device import atomic_json,file_hash

def main():
    protocol=json.loads((OUT/"protocol.json").read_text())
    segments=pd.read_csv(ROOT/"results/segments.csv")
    waves=np.load(ROOT/"data/cache/waves.npy",mmap_mode="r")
    rows=[];rng=np.random.default_rng(SEED)
    for fold in range(3):
        candidate=protocol["source_splits"][fold]["selected_candidate"]
        bundle=joblib.load(OUT/"models"/f"candidate{candidate}_fold{fold}.joblib")["bundle"]
        test=segments.loc[segments.fold_species.eq(fold)]
        positive=[]
        for label,subset in test.loc[test.label.ne("noise")].groupby("label"):
            positive.extend(rng.choice(subset.index,8,replace=False).tolist())
        negative=test.loc[test.label.eq("noise")]
        noise=[]
        for _ in positive:
            group=rng.choice(negative.group.unique())
            noise.append(rng.choice(negative.loc[negative.group.eq(group)].index))
        assert set(segments.iloc[positive].group).isdisjoint(protocol["source_splits"][fold]["fit_groups"])
        assert set(segments.iloc[noise].group).isdisjoint(protocol["source_splits"][fold]["fit_groups"])
        assert set(segments.iloc[noise].group).isdisjoint(protocol["source_splits"][fold]["calibration_groups"])
        signals=np.array(waves[positive],dtype="float32");background=np.array(waves[noise],dtype="float32")
        signals/=np.maximum(np.sqrt(np.mean(signals**2,axis=1,keepdims=True)),1e-8)
        background/=np.maximum(np.sqrt(np.mean(background**2,axis=1,keepdims=True)),1e-8)
        for kind,snr in [("Sem ruído adicionado",None),("Ruído real reservado",20),("Ruído real reservado",10),("Ruído real reservado",0)]:
            mixed=signals if snr is None else signals+background*10**(-snr/20)
            score=calibrated(bundle,frontend.features(mixed))
            for mode,threshold in [("Balanceado 0,5",.5),("Calibração FPR 5%",bundle["threshold_05"]),("Calibração FPR 10%",bundle["threshold_10"])]:
                detected=score>=threshold
                rows.append(dict(fold=fold,noise=kind,added_snr_db=snr,mode=mode,n_windows=len(positive),
                                 positive_window_recall=float(detected.mean()),positive_groups=segments.iloc[positive].group.nunique(),
                                 noise_groups=segments.iloc[noise].group.nunique()))
    pd.DataFrame(rows).to_csv(OUT/"noise_probe.csv",index=False)
    atomic_json(OUT/"noise_probe_manifest.json",dict(script_sha256=file_hash(ROOT/"probe_arduino_noise.py"),
                positive_label="recording-level mosquito, not event annotations",windows_per_species_per_fold=8,
                heldout_noise_sources_only=True,false_alarms_per_hour_measured=False))
    print(pd.DataFrame(rows).groupby(["mode","noise","added_snr_db"],dropna=False).positive_window_recall.mean())

if __name__=="__main__": main()
