"""Amendment A2 (exploratory): S4 synthetic deceptive-sound stress test and D1 mel-band occlusion.

Final models are refit on the whole development set with the hyper-parameters chosen
by inner grouped CV in E3 (read from E3 info.json). Synthetic signals never enter training.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reanalysis import frontend as fe
from reanalysis import train_v2 as tv

OUT = tv.OUT / "stress"


def synth_signals(seed=20261022, dur=5.0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * fe.SR)) / fe.SR
    sig = []
    def add(kind, f, x):
        x = x / np.max(np.abs(x)) * 0.1 + rng.normal(0, 1e-3, len(x))
        sig.append(dict(kind=kind, f_hz=f, x=x.astype("float32")))
    for f in range(300, 901, 50):
        jitter = 1 + 0.005 * np.sin(2 * np.pi * 3 * t + rng.uniform(0, 6.28))
        phase = 2 * np.pi * np.cumsum(f * jitter) / fe.SR
        add("pure_tone", f, np.sin(phase))
        add("harmonic_tone", f, sum(np.sin(k * phase) / k for k in range(1, 5)))
    for f in (60, 120):
        add("electrical_hum", f, sum(np.sin(2 * np.pi * k * f * t) / k for k in range(1, 8)))
    for _ in range(5):
        w = rng.normal(size=len(t)); spec = np.fft.rfft(w); fr = np.fft.rfftfreq(len(t), 1 / fe.SR)
        add("pink_noise", 0, np.fft.irfft(spec / np.sqrt(np.maximum(fr, 1)), len(t)))
        add("white_noise", 0, rng.normal(size=len(t)))
    return sig


def features(x):
    ws, _, _ = fe.windows(x)
    mels, tabs = [], []
    for w in ws:
        w = w / max(float(np.sqrt(np.mean(w ** 2))), 1e-12)
        m = fe.logmel(w); mels.append(m); tabs.append(np.r_[m.mean(1), m.std(1), fe.descriptors(w)])
    return np.stack(mels).astype("float32"), np.stack(tabs).astype("float32")


def final_models(man, win, mel, tab):
    info = {m: json.loads((tv.PRED / f"E3_external__{m}.info.json").read_text())["0"] for m in
            ["M1_logistic", "M2_histgb", "M3_cnn"]}
    files = tv.dev_files(man)
    w = tv.windows_of(win, files); wf = win.file_index.to_numpy()[w]; y = man.target.to_numpy()
    sw = tv.sample_weights(wf, y)
    models = {"M1_logistic": tv.fit_tabular("logistic", np.asarray(tab[w]), y[wf], sw, info["M1_logistic"]["chosen"]),
              "M2_histgb": tv.fit_tabular("histgb", np.asarray(tab[w]), y[wf], sw, info["M2_histgb"]["chosen"])}
    import torch
    torch.set_num_threads(8)
    pred_cnn, _ = tv.fit_cnn(np.asarray(mel[w]), y[wf], sw, tv.OUTER_SEED, info["M3_cnn"]["best_epoch"])
    models["M3_cnn"] = pred_cnn
    return models, w, wf


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man, win, mel, tab, _ = tv.load()
    models, w, wf = final_models(man, win, mel, tab)
    # S4 stress test
    rows = []
    for i, s in enumerate(synth_signals()):
        M, T = features(s["x"])
        for name, f in models.items():
            p = f(M if name == "M3_cnn" else T)
            agg = tv.aggregate(np.zeros(len(p), int), p, np.array([0]))[0]
            rows.append(dict(kind=s["kind"], f_hz=s["f_hz"], model=name, p0=agg[0], p1=agg[1], p2=agg[2], pred=int(agg.argmax())))
    S4 = pd.DataFrame(rows); S4.to_csv(OUT / "S4_synthetic_predictions.csv", index=False)
    summary = S4.groupby(["model", "kind"]).agg(n=("pred", "size"), called_aedes=("pred", lambda p: int((p == 2).sum())),
                                                called_other_mosquito=("pred", lambda p: int((p == 1).sum())),
                                                called_no_mosquito=("pred", lambda p: int((p == 0).sum())))
    summary.to_csv(OUT / "S4_summary.csv")
    print(summary.to_string())
    # D1 band occlusion (reliance, in-sample on development windows; not a performance estimate)
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(w), size=min(6000, len(w)), replace=False))
    Mw, Tw = np.asarray(mel[w[pick]]), np.asarray(tab[w[pick]])
    yw = man.target.to_numpy()[wf[pick]]
    centers = fe.mel_to_hz(np.linspace(fe.hz_to_mel(fe.FMIN), fe.hz_to_mel(fe.FMAX), fe.N_MELS + 2))[1:-1]
    tmean = Tw.mean(0)
    occl = []
    base = {n: f(Mw if n == "M3_cnn" else Tw)[:, 2] for n, f in models.items()}
    for g in range(8):
        bands = np.arange(5 * g, 5 * g + 5)
        M2 = Mw.copy(); M2[:, bands, :] = 0.0
        T2 = Tw.copy(); T2[:, bands] = tmean[bands]; T2[:, 40 + bands] = tmean[40 + bands]
        for n, f in models.items():
            p = f(M2 if n == "M3_cnn" else T2)[:, 2]
            occl.append(dict(model=n, band_group=g, f_lo=float(centers[bands[0]]), f_hi=float(centers[bands[-1]]),
                             mean_abs_delta_p_aedes=float(np.mean(np.abs(p - base[n]))),
                             delta_p_aedes_on_aedes=float(np.mean(p[yw == 2] - base[n][yw == 2]))))
    D1 = pd.DataFrame(occl); D1.to_csv(OUT / "D1_band_occlusion.csv", index=False)
    print(D1.round(3).to_string())


if __name__ == "__main__":
    main()
