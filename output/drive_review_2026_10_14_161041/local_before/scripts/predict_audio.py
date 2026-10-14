"""Diagnóstico, Segmentação Temporal de Enxames e Classificação Hierárquica de Áudio de Mosquitos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
import requests
from scipy.special import expit, logsumexp, softmax

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
# Execute the Drive version of robust_audio_features, while sharing the
# unchanged project frontend and frozen model artifacts.
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(1, str(ROOT))

import arduino_frontend
from arduino_mic_emulator import (
    TARGET_SR,
    emulate_arduino_mic,
    load_audio_via_ffmpeg,
)
from robust_audio_features import (
    analyze_harmonic_flutter,
)
from train_hierarchical_models import BRAZIL_PRIORS

MODELS_DIR = ROOT / "results/arduino_species/models"
PRESENCE_DIR = ROOT / "results/arduino/models"
WINDOW_SAMPLES = arduino_frontend.WINDOW_SAMPLES  # 15872 samples = 0.992 s @ 16 kHz

SPECIES_F0_PROFILES: dict[str, tuple[float, float]] = {
    "Aedes aegypti": (560.0, 90.0),
    "Aedes albopictus": (570.0, 90.0),
    "Aedes mediovittatus": (520.0, 85.0),
    "Aedes sierrensis": (460.0, 80.0),
    "Anopheles albimanus": (430.0, 85.0),
    "Anopheles arabiensis": (440.0, 85.0),
    "Anopheles atroparvus": (390.0, 75.0),
    "Anopheles dirus": (420.0, 80.0),
    "Anopheles farauti": (435.0, 85.0),
    "Anopheles freeborni": (385.0, 75.0),
    "Anopheles gambiae": (445.0, 85.0),
    "Anopheles merus": (430.0, 80.0),
    "Anopheles minimus": (450.0, 85.0),
    "Anopheles quadriannulatus": (425.0, 80.0),
    "Anopheles quadrimaculatus": (380.0, 75.0),
    "Anopheles stephensi": (440.0, 85.0),
    "Culex pipiens": (375.0, 65.0),
    "Culex quinquefasciatus": (395.0, 70.0),
    "Culex tarsalis": (360.0, 65.0),
    "Culiseta incidens": (310.0, 55.0),
}


def download_google_drive_file(url_or_id: str, destination: Path) -> Path:
    match = re.search(r"[-_\w]{25,}", url_or_id)
    if not match:
        raise ValueError(f"Não foi possível extrair ID do Google Drive de: {url_or_id}")
    file_id = match.group(0)
    session = requests.Session()
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = session.get(url, stream=True)
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            params = {"id": file_id, "confirm": value}
            response = session.get("https://drive.google.com/uc?export=download", params=params, stream=True)
            break
    response.raise_for_status()
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
    return destination


def marginalize_genus_power(
    species_probs: np.ndarray,
    genus_classes: np.ndarray,
    unique_genera: list[str],
    gamma: float = 1.0,
    tau: float = 1.5,
) -> dict[str, float]:
    logits = []
    for g in unique_genera:
        mask = genus_classes == g
        n_g = max(int(np.sum(mask)), 1)
        val = (
            logsumexp(tau * np.log(np.maximum(species_probs[mask], 1e-12)))
            - gamma * np.log(n_g)
        ) / tau
        logits.append(val)
    g_probs = softmax(np.asarray(logits, dtype=np.float64))
    return {g: float(p) for g, p in zip(unique_genera, g_probs)}


class HierarchicalMosquitoPipeline:
    def __init__(self, model_mode: str = "ensemble", region: str = "brazil"):
        self.model_mode = model_mode
        self.region = region.lower()
        self.models = self._load_models()
        self.classes = self.models[0]["classes"]
        self.num_classes = len(self.classes)
        self.genus_classes = np.array([c.split()[0] for c in self.classes])
        self.unique_genera = sorted(list(set(self.genus_classes)))

    def _load_models(self) -> list[dict]:
        loaded = []
        if self.model_mode == "deployed":
            fold_file = MODELS_DIR / "species_fold2.npz"
            loaded.append(self._parse_fold_file(fold_file))
        elif self.model_mode == "ensemble":
            for fold in [0, 2]:
                fold_file = MODELS_DIR / f"species_fold{fold}.npz"
                if fold_file.exists():
                    loaded.append(self._parse_fold_file(fold_file))
            if not loaded:
                for fold in [0, 1, 2]:
                    fold_file = MODELS_DIR / f"species_fold{fold}.npz"
                    if fold_file.exists():
                        loaded.append(self._parse_fold_file(fold_file))
        else:
            raise ValueError(f"Modo de modelo desconhecido: {self.model_mode}")
        return loaded

    @staticmethod
    def _parse_fold_file(path: Path) -> dict:
        data = np.load(path)
        classes = [
            c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in data["classes"]
        ]
        weights = []
        biases = []
        for i in range(10):
            w_key, b_key = f"weights{i}", f"bias{i}"
            if w_key in data and b_key in data:
                weights.append(data[w_key].astype("float32"))
                biases.append(data[b_key].astype("float32"))

        presence_w = [data["presence_weights0"].astype("float32")]
        presence_b = [data["presence_bias0"].astype("float32")]
        presence_out_w = data["presence_weights1"].astype("float32")
        presence_out_b = float(data["presence_bias1"])

        return {
            "classes": classes,
            "weights": weights,
            "biases": biases,
            "temperature": float(data["temperature"]),
            "confidence_threshold": float(data["confidence_threshold"]),
            "presence_threshold": float(data["presence_threshold"]),
            "presence_w": presence_w,
            "presence_b": presence_b,
            "presence_out_w": presence_out_w,
            "presence_out_b": presence_out_b,
            "presence_slope": float(data["presence_calibration_slope"]),
            "presence_intercept": float(data["presence_calibration_bias"]),
        }

    def predict_neural_presence(self, feat68: np.ndarray) -> np.ndarray:
        feat68 = np.atleast_2d(feat68).astype("float32")
        scores = []
        for m in self.models:
            h = np.maximum(0, feat68 @ m["presence_w"][0].T + m["presence_b"][0])
            raw = h @ m["presence_out_w"] + m["presence_out_b"]
            z = m["presence_slope"] * raw + m["presence_intercept"]
            scores.append(expit(z))
        return np.mean(scores, axis=0)

    def predict_species_probabilities(
        self,
        feat68: np.ndarray,
        flutter_info: dict | None = None,
    ) -> tuple[np.ndarray, float]:
        feat68 = np.atleast_2d(feat68).astype("float32")
        probs_list = []
        energies = []
        for m in self.models:
            x = feat68
            for i, (w, b) in enumerate(zip(m["weights"], m["biases"])):
                x = x @ w.T + b
                if i < len(m["weights"]) - 1:
                    x = np.maximum(0, x)
            T = m["temperature"]
            pz = softmax(x / T, axis=1)
            probs_list.append(pz)
            e = -T * logsumexp(x / T, axis=1)
            energies.append(e)

        mean_probs = np.mean(probs_list, axis=0)[0].astype(np.float64)
        mean_energy = float(np.mean(energies))
        ent = float(-np.sum(mean_probs * np.log(np.maximum(mean_probs, 1e-12))) / np.log(self.num_classes))
        low_peak_ratio_mean = float(feat68[0, 64])
        self.last_bio_fusion = dict(neural_entropy=ent, alpha_bio=0.0,
                                   low_peak_ratio_mean=low_peak_ratio_mean, bio_bonus_regime="none")

        if flutter_info is not None and flutter_info.get("f0_mean", 0.0) > 0:
            f0 = float(flutter_info["f0_mean"])
            f0_2 = float(flutter_info.get("second_f0_hz", 0.0))
            snr_aedes = float(flutter_info.get("aedes_snr_db", 0.0))
            is_swarm = bool(flutter_info.get("is_aedes_swarm", False))
            h2_h1 = float(flutter_info.get("h2_h1_ratio", 0.0))

            log_bio = np.zeros(self.num_classes, dtype=np.float64)
            for idx, sp in enumerate(self.classes):
                mu, sigma = SPECIES_F0_PROFILES.get(sp, (440.0, 90.0))
                z1 = -0.5 * ((f0 - mu) / sigma) ** 2
                if f0_2 >= 350.0:
                    z2 = -0.5 * ((f0_2 - mu) / sigma) ** 2
                    z_f0 = float(np.logaddexp(z1, z2) - np.log(2.0))
                else:
                    z_f0 = float(z1)

                gen = sp.split()[0]
                bonus = 0.0
                if (gen == "Aedes" and ent >= 0.73 and low_peak_ratio_mean < 0.20
                        and 420.0 <= f0 <= 740.0 and snr_aedes >= 0.8):
                    bonus += 0.80 * min(max(snr_aedes / 5.0, 0.0), 2.2)
                    if is_swarm:
                        bonus += 0.70
                    if h2_h1 >= 0.12:
                        bonus += 0.35
                    if sp == "Aedes aegypti":
                        bonus += 0.45
                    self.last_bio_fusion["bio_bonus_regime"] = "Aedes"
                # S7 in the paired D3 manifest is Culex pipiens, NOT array index 7.
                if (sp == "Culex pipiens" and ent >= 0.73
                        and low_peak_ratio_mean >= 0.20 and h2_h1 < 0.90):
                    bonus += 0.80
                    self.last_bio_fusion["bio_bonus_regime"] = "S7_Culex_pipiens"
                log_bio[idx] = z_f0 + bonus

            bio_probs = softmax(log_bio)
            alpha_bio = float(np.clip((ent - 0.68) / 0.20, 0.05, 0.70))
            if is_swarm and ent >= 0.74:
                alpha_bio = 0.65
            self.last_bio_fusion["alpha_bio"] = alpha_bio
            log_fused = (1.0 - alpha_bio) * np.log(np.maximum(mean_probs, 1e-12)) + alpha_bio * np.log(
                np.maximum(bio_probs, 1e-12)
            )
            mean_probs = softmax(log_fused)

        if self.region == "brazil":
            probs_adj = np.zeros_like(mean_probs)
            for i, sp in enumerate(self.classes):
                gen = sp.split()[0]
                prior = BRAZIL_PRIORS.get(gen, {}).get(sp, 0.05)
                probs_adj[i] = mean_probs[i] * prior
            probs_adj /= max(float(probs_adj.sum()), 1e-12)
            mean_probs = probs_adj

        return mean_probs.astype("float32"), mean_energy

    def evaluate_windows(
        self,
        features: np.ndarray,
        windows_pcm: list[np.ndarray],
        presence_threshold: float | None = None,
        confidence_threshold: float | None = None,
        energy_threshold: float = -2.5,
    ) -> list[dict]:
        features = np.asarray(features, dtype="float32")
        n_windows = len(features)
        results = []

        if presence_threshold is None:
            presence_threshold = 0.50
        if confidence_threshold is None:
            confidence_threshold = 0.28

        neural_presence_probs = self.predict_neural_presence(features)

        genus_history: list[str] = []
        species_history: list[str] = []

        for i in range(n_windows):
            feat = features[i : i + 1]
            p_neural = float(neural_presence_probs[i])
            w_pcm = windows_pcm[i]

            flutter_info = analyze_harmonic_flutter(w_pcm, sr=TARGET_SR)
            is_60hz = flutter_info["is_electrical_60hz"]
            is_clipped = flutter_info["is_clipped"]
            is_bio_flight = flutter_info["is_biological_flutter"]
            is_aedes_swarm = flutter_info["is_aedes_swarm"]
            f0_mean = flutter_info["f0_mean"]
            f0_std = flutter_info["f0_std"]
            f0_iqr = flutter_info["f0_iqr"]
            second_f0 = flutter_info["second_f0_hz"]
            h_ratio = flutter_info["harmonic_ratio"]
            h2_h1 = flutter_info["h2_h1_ratio"]
            aedes_snr_db = flutter_info["aedes_snr_db"]
            comb_strength = flutter_info["comb_strength"]
            active_duty = flutter_info["active_duty_cycle"]
            swarm_peaks = flutter_info["swarm_peak_count"]
            swarm_score = flutter_info["swarm_score"]
            regime = flutter_info["regime"]

            if regime == "SILENCIO_DIGITAL" or is_clipped:
                p_pres = 0.0
                is_detected = False
            elif is_60hz:
                p_pres = round(0.15 * p_neural, 4)
                is_detected = False
            else:
                p_pres = float(0.75 * swarm_score + 0.25 * p_neural) if is_bio_flight else float(
                    min(0.45, 0.65 * swarm_score + 0.35 * p_neural)
                )
                is_detected = bool(is_bio_flight or (p_pres >= presence_threshold and comb_strength >= 0.50))

            if is_clipped:
                status = "SATURAÇÃO / CLIPPING DIGITAL (DESCARTADO)"
            elif regime == "SILENCIO_DIGITAL":
                status = "AUSENTE (SILÊNCIO DIGITAL)"
            elif is_60hz:
                status = f"RUÍDO ELÉTRICO/ESTACIONÁRIO ({f0_mean:.0f} Hz, harmônico 60Hz)"
            elif not is_detected:
                status = "AUSENTE (RUÍDO DE FUNDO)"
            else:
                status = "CANDIDATO"

            if is_detected:
                sp_probs, free_energy = self.predict_species_probabilities(feat, flutter_info=flutter_info)
                is_ood = bool(free_energy > energy_threshold and not is_aedes_swarm)

                genus_probs = marginalize_genus_power(
                    sp_probs, self.genus_classes, self.unique_genera, gamma=1.0, tau=1.5
                )
                top_genus = max(genus_probs, key=genus_probs.get)
                genus_conf = genus_probs[top_genus]

                sp_idx = int(np.argmax(sp_probs))
                cand_species = self.classes[sp_idx]
                sp_conf = float(sp_probs[sp_idx])

                accepted = bool((sp_conf >= confidence_threshold or is_aedes_swarm) and not is_ood)

                genus_history.append(top_genus)
                genus_history = genus_history[-3:]
                species_history.append(cand_species if accepted else "---")
                species_history = species_history[-3:]

                confirmed_genus = None
                if len(genus_history) >= 2 and genus_history.count(top_genus) >= 2:
                    confirmed_genus = top_genus

                confirmed_species = None
                if (
                    len(species_history) >= 2
                    and cand_species != "---"
                    and species_history.count(cand_species) >= 2
                ):
                    confirmed_species = cand_species

                mode_tag = "ENXAME" if is_aedes_swarm else "VOO ISOLADO"
                if is_ood:
                    status = "INCERTO (FORA DA DISTRIBUIÇÃO)"
                elif confirmed_genus and confirmed_species:
                    status = f"CONFIRMADO [{mode_tag}] ({confirmed_species})"
                elif confirmed_genus:
                    status = f"GÊNERO CONFIRMADO [{mode_tag}] ({confirmed_genus})"
                elif accepted:
                    status = f"CANDIDATO [{mode_tag}] ({top_genus}: {cand_species})"
                else:
                    status = f"INCERTO ({top_genus}, baixa conf)"
            else:
                sp_probs = np.zeros(self.num_classes, dtype="float32")
                genus_probs = {g: 0.0 for g in self.unique_genera}
                top_genus = None
                genus_conf = 0.0
                cand_species = None
                sp_conf = 0.0
                free_energy = 0.0
                confirmed_genus = None
                confirmed_species = None
                genus_history.append("---")
                genus_history = genus_history[-3:]
                species_history.append("---")
                species_history = species_history[-3:]

            results.append(
                {
                    "window_index": i,
                    "presence_score": round(p_pres, 4),
                    "neural_presence_score": round(p_neural, 4),
                    "swarm_score": round(swarm_score, 4),
                    "detected": is_detected,
                    "is_aedes_swarm": is_aedes_swarm,
                    "regime": regime,
                    "f0_hz": f0_mean,
                    "second_f0_hz": second_f0,
                    "f0_std": f0_std,
                    "f0_iqr": f0_iqr,
                    "harmonic_ratio": h_ratio,
                    "h2_h1_ratio": h2_h1,
                    "aedes_snr_db": aedes_snr_db,
                    "comb_strength": comb_strength,
                    "active_duty_cycle": active_duty,
                    "swarm_peak_count": swarm_peaks,
                    "is_electrical_60hz": is_60hz,
                    "is_clipped": is_clipped,
                    "genus_candidate": top_genus,
                    "genus_confidence": round(genus_conf, 4),
                    "confirmed_genus": confirmed_genus,
                    "candidate_species": cand_species,
                    "species_confidence": round(sp_conf, 4),
                    "free_energy": round(free_energy, 2),
                    "confirmed_species": confirmed_species,
                    "status": status,
                    "genus_probabilities": {k: round(v, 4) for k, v in genus_probs.items()},
                    "probabilities": [round(float(p), 4) for p in sp_probs],
                    "neural_entropy": self.last_bio_fusion["neural_entropy"] if is_detected else None,
                    "alpha_bio": self.last_bio_fusion["alpha_bio"] if is_detected else 0.0,
                    "low_peak_ratio_mean": float(feat[0, 64]),
                    "bio_bonus_regime": self.last_bio_fusion["bio_bonus_regime"] if is_detected else "none",
                }
            )

        return results


def extract_temporal_intervals(df: pd.DataFrame) -> list[dict]:
    if len(df) == 0:
        return []

    labels = []
    for _, r in df.iterrows():
        if r.get("is_clipped", False):
            labels.append("ARTEFATO_CLIPPING_SATURACAO")
        elif r["is_aedes_swarm"] or (
            r["detected"] and r["genus_candidate"] == "Aedes" and r["aedes_snr_db"] >= 1.0
        ):
            labels.append("ENXAME_AEDES_PRESENTE")
        elif r["detected"]:
            labels.append("PASSAGEM_CURTA_ISOLADA")
        else:
            labels.append("AUSENTE_RUIDO_FUNDO")

    smoothed = list(labels)
    for i in range(1, len(smoothed) - 1):
        if (
            smoothed[i] != "ARTEFATO_CLIPPING_SATURACAO"
            and smoothed[i - 1] == "ENXAME_AEDES_PRESENTE"
            and smoothed[i + 1] == "ENXAME_AEDES_PRESENTE"
        ):
            smoothed[i] = "ENXAME_AEDES_PRESENTE"

    intervals = []
    cur_label = smoothed[0]
    start_idx = 0

    for i in range(1, len(smoothed) + 1):
        if i == len(smoothed) or smoothed[i] != cur_label:
            sub = df.iloc[start_idx:i]
            t_start = float(sub["start_s"].iloc[0])
            t_end = float(sub["end_s"].iloc[-1])
            dur = round(t_end - t_start, 2)
            active_sub = sub[sub["detected"]]
            f0_med = float(active_sub["f0_hz"].median()) if len(active_sub) else float(sub["f0_hz"].median())
            snr_med = float(sub["aedes_snr_db"].median())
            score_mean = float(sub["presence_score"].mean())
            top_sp = (
                active_sub["candidate_species"].mode().iloc[0]
                if len(active_sub) and active_sub["candidate_species"].notna().any()
                else None
            )
            intervals.append(
                {
                    "segment_type": cur_label,
                    "aedes_present": cur_label in ("ENXAME_AEDES_PRESENTE", "PASSAGEM_CURTA_ISOLADA"),
                    "is_sustained_swarm": bool(cur_label == "ENXAME_AEDES_PRESENTE" and dur >= 2.0),
                    "start_s": round(t_start, 2),
                    "end_s": round(t_end, 2),
                    "duration_s": dur,
                    "windows_count": len(sub),
                    "median_f0_hz": round(f0_med, 1) if f0_med > 0 else 0.0,
                    "median_aedes_snr_db": round(snr_med, 2),
                    "mean_presence_score": round(score_mean, 4),
                    "dominant_species": top_sp,
                }
            )
            if i < len(smoothed):
                cur_label = smoothed[i]
                start_idx = i

    return intervals


def process_audio_file(
    audio_path: str | Path,
    emulate_mic: bool = False,
    hop_seconds: float = 0.5,
    classifier: HierarchicalMosquitoPipeline | None = None,
    presence_threshold: float | None = None,
    confidence_threshold: float | None = None,
    energy_threshold: float = -2.5,
) -> tuple[pd.DataFrame, dict]:
    if classifier is None:
        classifier = HierarchicalMosquitoPipeline(model_mode="ensemble", region="brazil")

    raw_audio, sr = load_audio_via_ffmpeg(audio_path, target_sr=TARGET_SR)
    raw_audio = np.asarray(raw_audio, dtype=np.float32)
    if raw_audio.ndim > 1:
        raw_audio = np.mean(raw_audio, axis=1)
    if sr != TARGET_SR:
        from scipy import signal as _sig
        gcd = int(np.gcd(int(sr), int(TARGET_SR)))
        raw_audio = _sig.resample_poly(raw_audio, int(TARGET_SR) // gcd, int(sr) // gcd).astype(np.float32)
        sr = TARGET_SR
    duration = len(raw_audio) / float(TARGET_SR)

    if emulate_mic:
        processed_pcm = emulate_arduino_mic(
            raw_audio, sr=TARGET_SR, inject_noise=True, quantize=True
        ).ravel()
    else:
        processed_pcm = np.asarray(arduino_frontend.pcm16(raw_audio), dtype=np.int16).ravel()

    hop_samples = int(round(hop_seconds * TARGET_SR))
    total_samples = len(processed_pcm)

    windows_raw = []
    starts_s = []
    ends_s = []

    pos = 0
    while pos + WINDOW_SAMPLES <= total_samples:
        window = processed_pcm[pos : pos + WINDOW_SAMPLES]
        windows_raw.append(window)
        starts_s.append(pos / float(TARGET_SR))
        ends_s.append((pos + WINDOW_SAMPLES) / float(TARGET_SR))
        pos += hop_samples

    if len(windows_raw) == 0:
        raise ValueError(f"Áudio com duração ({duration:.2f} s) menor que a janela mínima (0.992 s).")

    feats = arduino_frontend.features(np.stack(windows_raw), replay_pcm=True)

    eval_rows = classifier.evaluate_windows(
        feats,
        windows_pcm=windows_raw,
        presence_threshold=presence_threshold,
        confidence_threshold=confidence_threshold,
        energy_threshold=energy_threshold,
    )

    for i, row in enumerate(eval_rows):
        row["start_s"] = round(starts_s[i], 3)
        row["end_s"] = round(ends_s[i], 3)

    df = pd.DataFrame(eval_rows)
    intervals = extract_temporal_intervals(df)
    # Non-overlapping decision timeline at the hop grid. Keep the original
    # overlapping analysis intervals separately for compatibility.
    swarm_timeline = []
    for i, interval in enumerate(intervals):
        start = float(interval["start_s"])
        end = float(intervals[i+1]["start_s"]) if i+1 < len(intervals) else float(df.end_s.iloc[-1])
        sub = df[(df.start_s >= start) & (df.start_s < end)]
        swarm_timeline.append(dict(start_s=start, end_s=end, duration_s=round(end-start, 6),
            swarm_present=interval["segment_type"] == "ENXAME_AEDES_PRESENTE",
            median_f0_hz=interval["median_f0_hz"],
            mean_aedes_snr_db=float(sub.aedes_snr_db.mean()),
            segment_type=interval["segment_type"]))

    detected_mask = df.detected.to_numpy()
    swarm_mask = df.is_aedes_swarm.to_numpy()
    total_windows = len(df)
    detected_windows = int(np.sum(detected_mask))
    swarm_windows = int(np.sum(swarm_mask))
    active_ratio = detected_windows / float(total_windows)
    swarm_ratio = swarm_windows / float(total_windows)

    pos_df = df[df.detected]
    genus_counts = pos_df.genus_candidate.value_counts().to_dict() if len(pos_df) else {}
    species_counts = pos_df.candidate_species.value_counts().to_dict() if len(pos_df) else {}
    confirmed_genus_counts = (
        df.confirmed_genus.dropna().value_counts().to_dict()
        if df.confirmed_genus.notna().any()
        else {}
    )
    confirmed_sp_counts = (
        df.confirmed_species.dropna().value_counts().to_dict()
        if df.confirmed_species.notna().any()
        else {}
    )

    top_genus = max(genus_counts, key=genus_counts.get) if genus_counts else None
    top_confirmed_genus = (
        max(confirmed_genus_counts, key=confirmed_genus_counts.get)
        if confirmed_genus_counts
        else None
    )
    top_species = max(species_counts, key=species_counts.get) if species_counts else None
    top_confirmed_sp = (
        max(confirmed_sp_counts, key=confirmed_sp_counts.get) if confirmed_sp_counts else None
    )

    flight_f0 = float(pos_df.f0_hz.median()) if len(pos_df) else 0.0
    flight_f0_iqr = (
        float(np.percentile(pos_df.f0_hz, 75) - np.percentile(pos_df.f0_hz, 25))
        if len(pos_df) >= 2
        else 0.0
    )

    summary = {
        "audio_file": str(audio_path),
        "duration_seconds": round(duration, 2),
        "emulate_arduino_mic": emulate_mic,
        "region_prior": classifier.region,
        "total_windows": total_windows,
        "detected_windows": detected_windows,
        "aedes_swarm_windows": swarm_windows,
        "active_ratio_pct": round(active_ratio * 100, 2),
        "swarm_ratio_pct": round(swarm_ratio * 100, 2),
        "mosquito_present": bool(active_ratio >= 0.10 or detected_windows >= 3),
        "aedes_swarm_detected": bool(swarm_windows >= 3),
        "median_flight_f0_hz": round(flight_f0, 1),
        "iqr_flight_f0_hz": round(flight_f0_iqr, 1),
        "primary_genus": top_genus,
        "confirmed_genus": top_confirmed_genus,
        "primary_species": top_species,
        "confirmed_species": top_confirmed_sp,
        "genus_distribution": genus_counts,
        "species_distribution": species_counts,
        "temporal_intervals": intervals,
        "swarm_timeline": swarm_timeline,
        "swarm_active_intervals": [i for i in swarm_timeline if i["swarm_present"]],
        "timeline_resolution_s": hop_seconds,
        "analysis_window_s": WINDOW_SAMPLES / float(TARGET_SR),
        "unanalysed_tail_s": round(duration-float(df.end_s.iloc[-1]), 6),
        "timeline_interpretation": "Estimated offline classifier decisions, not entomological ground truth; interval smoothing uses the next window.",
    }

    return df, summary


def print_cli_report(summary: dict, df: pd.DataFrame):
    print("\n" + "=" * 82)
    print(" RELATÓRIO HIERÁRQUICO E SEGMENTAÇÃO TEMPORAL DE ENXAME DE AEDES (ML + SHS) ")
    print("=" * 82)
    print(f"Arquivo:               {summary['audio_file']}")
    print(f"Duração Total:         {summary['duration_seconds']} s")
    print(
        f"Emulação Mic Arduino:  {'ATIVADA (ST MP34DT05)' if summary['emulate_arduino_mic'] else 'DESATIVADA (Áudio Bruto)'}"
    )
    print(f"Prior Epidemiológico:  {summary['region_prior'].upper()}")
    print(
        f"Janelas de Voo Ativo:  {summary['detected_windows']}/{summary['total_windows']} ({summary['active_ratio_pct']}%)"
    )
    print(
        f"Janelas de Enxame:     {summary['aedes_swarm_windows']}/{summary['total_windows']} ({summary['swarm_ratio_pct']}%)"
    )
    print(
        f"Diagnóstico Tier 1:    {'ENXAME DE AEDES DETECTADO' if summary['aedes_swarm_detected'] else ('MOSQUITO DETECTADO' if summary['mosquito_present'] else 'NENHUM MOSQUITO IDENTIFICADO')}"
    )

    if summary["mosquito_present"]:
        print(
            f"Fundamental Mediana:   F0 = {summary['median_flight_f0_hz']:.1f} Hz (IQR = {summary['iqr_flight_f0_hz']:.1f} Hz)"
        )
        print("\n--- TIER 2: VETOR EPIDEMIOLÓGICO (GÊNERO COM CORREÇÃO DE CARDINALIDADE) ---")
        gen_str = summary["confirmed_genus"] or summary["primary_genus"]
        print(
            f"Gênero Identificado:   {gen_str} (Confirmação Causal: {summary['confirmed_genus'] is not None})"
        )
        for g, count in summary["genus_distribution"].items():
            pct = count / summary["detected_windows"] * 100
            print(f"  * {g:<14}: {count:3d} janelas ({pct:5.1f}%)")

        print("\n--- TIER 3: ESPÉCIE ESPECÍFICA (FUSÃO HARMÔNICA + NEURAL) ---")
        sp_str = summary["confirmed_species"] or summary["primary_species"]
        print(f"Espécie Confirmada:    {sp_str}")
        for sp, count in list(summary["species_distribution"].items())[:5]:
            pct = count / summary["detected_windows"] * 100
            print(f"  * {sp:<26}: {count:3d} janelas ({pct:5.1f}%)")

    print("\n--- MAPA DE INTERVALOS TEMPORAIS (QUANDO O ENXAME APARECE vs QUANDO NÃO APARECE) ---")
    print(
        f"{'Intervalo (s)':<16} | {'Duração':<8} | {'Status / Regime':<28} | {'F0 Med':<8} | {'SNR Aedes':<10} | {'Espécie Dominante'}"
    )
    print("-" * 98)
    for seg in summary["temporal_intervals"]:
        t_range = f"{seg['start_s']:5.1f} - {seg['end_s']:5.1f}s"
        dur_str = f"{seg['duration_s']:5.1f}s"
        f0_str = f"{seg['median_f0_hz']:5.1f}Hz" if seg["aedes_present"] else "  ---  "
        snr_str = f"{seg['median_aedes_snr_db']:+5.1f} dB"
        sp_dom = str(seg["dominant_species"] or "---")
        print(
            f"{t_range:<16} | {dur_str:<8} | {seg['segment_type']:<28} | {f0_str:<8} | {snr_str:<10} | {sp_dom}"
        )
    print("=" * 82 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnóstico Hierárquico e Segmentação Temporal de Enxame de Aedes via ML + SHS."
    )
    parser.add_argument(
        "audio", nargs="?", type=str, help="Caminho do áudio local (.wav, .mp3) ou link/ID do Google Drive"
    )
    parser.add_argument("--audio", dest="audio_option", type=str, help="Alias para o áudio de entrada")
    parser.add_argument(
        "--emulate-arduino", action="store_true", help="Aplica emulação acústica do mic MP34DT05"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Executa comparação lado a lado (Bruto vs Mic Arduino)",
    )
    parser.add_argument(
        "--hop", type=float, default=0.5, help="Passo temporal em segundos (padrão: 0.5s)"
    )
    parser.add_argument(
        "--model",
        choices=["deployed", "ensemble"],
        default="ensemble",
        help="Modo do modelo (padrão: ensemble)",
    )
    parser.add_argument(
        "--region",
        choices=["brazil", "global"],
        default="brazil",
        help="Prior biogeográfico regional (padrão: brazil)",
    )
    parser.add_argument("--presence-threshold", type=float, default=None, help="Limiar de presença")
    parser.add_argument(
        "--confidence-threshold", type=float, default=None, help="Limiar de confiança"
    )
    parser.add_argument(
        "--output-csv", type=str, default=None, help="Caminho para salvar o CSV detalhado"
    )
    parser.add_argument(
        "--output-json", "--output-summary", dest="output_json", type=str, default=None, help="Caminho para salvar o resumo em JSON"
    )

    args = parser.parse_args()

    audio_arg = args.audio_option or args.audio
    if not audio_arg:
        parser.error("Informe --audio ou o argumento posicional audio.")
    temp_download = None

    if (
        "drive.google.com" in audio_arg
        or len(audio_arg) > 25
        and "/" not in audio_arg
        and not Path(audio_arg).exists()
    ):
        temp_download = ROOT / "downloaded_drive_audio.wav"
        audio_path = download_google_drive_file(audio_arg, temp_download)
    else:
        audio_path = Path(audio_arg)
        if not audio_path.exists():
            print(f"Erro: Arquivo '{audio_path}' não encontrado.", file=sys.stderr)
            sys.exit(1)

    try:
        classifier = HierarchicalMosquitoPipeline(model_mode=args.model, region=args.region)

        if args.compare:
            print(
                "\n[MODO DE COMPARAÇÃO ATIVADO: ÁUDIO BRUTO vs MICROFONE ARDUINO NANO 33 BLE SENSE]"
            )
            df_raw, sum_raw = process_audio_file(
                audio_path,
                emulate_mic=False,
                hop_seconds=args.hop,
                classifier=classifier,
                presence_threshold=args.presence_threshold,
                confidence_threshold=args.confidence_threshold,
            )
            df_emu, sum_emu = process_audio_file(
                audio_path,
                emulate_mic=True,
                hop_seconds=args.hop,
                classifier=classifier,
                presence_threshold=args.presence_threshold,
                confidence_threshold=args.confidence_threshold,
            )

            print_cli_report(sum_raw, df_raw)
            print_cli_report(sum_emu, df_emu)

            df_to_save, sum_to_save = df_emu, sum_emu
        else:
            df_to_save, sum_to_save = process_audio_file(
                audio_path,
                emulate_mic=args.emulate_arduino,
                hop_seconds=args.hop,
                classifier=classifier,
                presence_threshold=args.presence_threshold,
                confidence_threshold=args.confidence_threshold,
            )
            print_cli_report(sum_to_save, df_to_save)

        if args.output_csv:
            out_csv = Path(args.output_csv)
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            df_to_save.to_csv(out_csv, index=False)
            print(f"-> Relatório detalhado salvo em CSV: {out_csv}")

        if args.output_json:
            out_json = Path(args.output_json)
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(sum_to_save, indent=2, ensure_ascii=False))
            print(f"-> Resumo analítico salvo em JSON: {out_json}")

    finally:
        if temp_download is not None and temp_download.exists():
            temp_download.unlink()


if __name__ == "__main__":
    main()
