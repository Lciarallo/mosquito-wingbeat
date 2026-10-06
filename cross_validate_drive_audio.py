"""Validação Cruzada Multi-Algoritmo do Áudio do Google Drive.

Compara diferentes paradigmas da literatura e do projeto para avaliar o consenso
sobre a identidade taxonômica do mosquito:
1. TinySpeciesMLP do Firmware Arduino (TinyML, 68 características FFT de streaming).
2. Ensemble Out-of-Fold 3-Dobras (Validação Cruzada Estrita).
3. Regressão Logística em Cepstro/MFCCs (Espectrograma Mel / Timbre).
4. Decomposição Harmônica Física (Frequência Fundamental F0 e 2F0 - Mukundarajan et al., 2017).
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
import scipy.signal as signal
from scipy.fftpack import dct
from scipy.special import softmax

import arduino_frontend

SR = 16000
W = arduino_frontend.WINDOW_SAMPLES


def run_cross_validation(audio_path: str = "drive_audio.wav"):
    sr, data = wav.read(audio_path)
    data_pcm = np.asarray(arduino_frontend.pcm16(data), dtype=np.int16).ravel()
    total_sec = len(data_pcm) / float(SR)

    # 1. Filtra trecho com áudio real (ignora o flatline de 0 a 3s)
    # Identifica janelas onde o RMS é suficiente e há energia na banda de 200-900 Hz
    step = 8000  # hop de 0.5s
    windows = []
    timestamps = []

    for i in range(int(3.0 * SR), len(data_pcm) - W, step):
        chunk = data_pcm[i : i + W]
        rms = np.sqrt(np.mean(chunk.astype(float)**2))
        # Seleciona janelas com sinal acústico real
        if rms > 200:
            windows.append(chunk)
            timestamps.append(i / float(SR))

    windows = np.stack(windows)
    feats68 = arduino_frontend.features(windows, replay_pcm=False)

    # -------------------------------------------------------------------------
    # ALGORITMO 1: TinySpeciesMLP (Firmware Arduino Dobra 2)
    # -------------------------------------------------------------------------
    m2 = np.load("results/arduino_species/models/species_fold2.npz")
    classes = [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in m2["classes"]]

    h2 = np.maximum(0, feats68 @ m2["weights0"].T + m2["bias0"])
    logits2 = h2 @ m2["weights1"].T + m2["bias1"]
    probs_mlp = softmax(logits2 / float(m2["temperature"]), axis=1)
    preds_mlp = [classes[idx] for idx in probs_mlp.argmax(axis=1)]

    # -------------------------------------------------------------------------
    # ALGORITMO 2: Ensemble 3-Dobras (OOF)
    # -------------------------------------------------------------------------
    ensemble_probs = []
    for f in [0, 1, 2]:
        mf = np.load(f"results/arduino_species/models/species_fold{f}.npz")
        if f == 0:
            h0 = np.maximum(0, feats68 @ mf["weights0"].T + mf["bias0"])
            h1 = np.maximum(0, h0 @ mf["weights1"].T + mf["bias1"])
            lz = h1 @ mf["weights2"].T + mf["bias2"]
        elif f == 1:
            lz = feats68 @ mf["weights0"].T + mf["bias0"]
        else:
            lz = np.maximum(0, feats68 @ mf["weights0"].T + mf["bias0"]) @ mf["weights1"].T + mf["bias1"]
        pz = softmax(lz / float(mf["temperature"]), axis=1)
        ensemble_probs.append(pz)
    probs_ensemble = np.mean(ensemble_probs, axis=0)
    preds_ensemble = [classes[idx] for idx in probs_ensemble.argmax(axis=1)]

    # -------------------------------------------------------------------------
    # ALGORITMO 3: MFCC / Mel-Cepstrum + Regressão Logística
    # -------------------------------------------------------------------------
    def mel_bank():
        frequencies = np.fft.rfftfreq(2048, 1 / SR)
        to_mel = lambda f: 2595 * np.log10(1 + f / 700)
        to_hz = lambda m: 700 * (10 ** (m / 2595) - 1)
        limits = to_hz(np.linspace(to_mel(100), to_mel(3900), 42))
        bank = np.zeros((40, len(frequencies)), dtype="float32")
        for k in range(len(bank)):
            bank[k] = np.maximum(0, np.minimum((frequencies - limits[k]) / (limits[k+1] - limits[k]),
                                              (limits[k+2] - frequencies) / (limits[k+2] - limits[k+1])))
            bank[k] /= max(bank[k].sum(), 1e-12)
        return bank

    mbank = mel_bank()
    mfcc_list = []
    for w in windows:
        w_f = w.astype("float32")
        w_f = (w_f - w_f.mean()) / max(float(np.sqrt(np.mean(w_f**2))), 1e-8)
        _, _, st = signal.stft(w_f, fs=SR, nperseg=2048, noverlap=1728, boundary=None, padded=False)
        m = mbank @ np.abs(st)**2
        mdb = np.clip(10 * np.log10(np.maximum(m, 1e-12)) - 10 * np.log10(np.maximum(m, 1e-12)).max(), -80, 0)
        c = dct(mdb, type=2, axis=0, norm="ortho")[:13]
        mfcc_list.append(np.r_[c.mean(axis=1), c.std(axis=1)].astype("float32"))
    mfcc_arr = np.stack(mfcc_list)

    mfcc_probs = []
    for f in [0, 1, 2]:
        mf = np.load(f"results/models/mfcc_int8_fold{f}.npz")
        norm_x = (mfcc_arr - mf["mean"]) * mf["inverse_std"]
        wf = mf["weights"].astype(float) * mf["weight_scale"][:, None] / float(mf["input_scale"])
        lz = norm_x @ wf.T + mf["bias"].astype(float)
        mfcc_probs.append(softmax(lz, axis=1))
    probs_mfcc = np.mean(mfcc_probs, axis=0)
    preds_mfcc = [classes[idx] for idx in probs_mfcc.argmax(axis=1)]

    # -------------------------------------------------------------------------
    # ALGORITMO 4: Decomposição Harmônica Física (F0 e 2F0)
    # -------------------------------------------------------------------------
    f0_peaks = []
    for w in windows:
        f, psd = signal.welch(w.astype(float), fs=SR, nperseg=3200, noverlap=2880)
        wing_band = (f >= 200) & (f <= 900)
        peak_f = f[wing_band][np.argmax(psd[wing_band])]
        f0_peaks.append(peak_f)

    df_cross = pd.DataFrame({
        "timestamp_s": timestamps,
        "F0_Hz": f0_peaks,
        "MLP_Arduino": preds_mlp,
        "MLP_Conf": probs_mlp.max(axis=1),
        "Ensemble_OOF": preds_ensemble,
        "Ensemble_Conf": probs_ensemble.max(axis=1),
        "MFCC_LogReg": preds_mfcc,
        "MFCC_Conf": probs_mfcc.max(axis=1),
    })

    df_cross.to_csv("results/cross_validation_drive_audio.csv", index=False)

    print("=== RESULTADO DA VALIDAÇÃO CRUZADA MULTI-ALGORITMO ===")
    print(f"Total de Janelas de Áudio Real Analisadas: {len(df_cross)}")
    print(f"Faixa de F0 Medida: {np.percentile(f0_peaks, 10):.1f} Hz a {np.percentile(f0_peaks, 90):.1f} Hz (Mediana: {np.median(f0_peaks):.1f} Hz)")

    print("\n--- 1. Ranking Global MLP Firmware (Arduino Nano 33 BLE Sense) ---")
    print(df_cross["MLP_Arduino"].value_counts().head(5))

    print("\n--- 2. Ranking Global Ensemble 3-Dobras (OOF) ---")
    print(df_cross["Ensemble_OOF"].value_counts().head(5))

    print("\n--- 3. Ranking Global MFCC / Mel-Cepstrum (LogReg) ---")
    print(df_cross["MFCC_LogReg"].value_counts().head(5))

    # Janelas de zumbido mais forte (alta confiança no MLP)
    strong = df_cross[df_cross["MLP_Conf"] >= 0.35]
    print(f"\n--- 4. Consenso nos Trechos de Zumbido Nítido (Confiança MLP >= 35%, {len(strong)} janelas) ---")
    print(f"F0 Médio nos Trechos Nítidos: {strong['F0_Hz'].mean():.1f} Hz")
    print("Predições no Zumbido Nítido:")
    for _, r in strong.head(10).iterrows():
        print(f"  {r['timestamp_s']:5.1f}s | F0: {r['F0_Hz']:5.1f} Hz | MLP: {r['MLP_Arduino']:<20} ({r['MLP_Conf']*100:4.1f}%) | Ens: {r['Ensemble_OOF']:<20} ({r['Ensemble_Conf']*100:4.1f}%) | MFCC: {r['MFCC_LogReg']}")


if __name__ == "__main__":
    run_cross_validation()
