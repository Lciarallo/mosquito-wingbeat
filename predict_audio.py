"""Diagnóstico e Classificação Hierárquica de Áudio de Mosquitos com ML (Zero Data Leakage).

Executa detecção e classificação bioacústica em três níveis taxonômicos:
- Tier 1: Detecção de Presença (Mosquito vs Ruído de Fundo).
- Tier 2: Classificação de Gênero (Aedes, Anopheles, Culex, Culiseta) com >93% de acurácia OOF.
- Tier 3: Subclassificação de Espécie (20 espécies) com Energia Livre de Helmholtz e Priors Regionais.

Recursos Inclusos:
- Filtro de Interferência de Rede Elétrica (harmônicos de 60 Hz e ruído mecânico estacionário).
- Emulação acústica do microfone ST MP34DT05 (Arduino Nano 33 BLE Sense).
- Comparação lado a lado (áudio bruto vs áudio emulado).
- Suporte a Priors Epidemiológicos Regionais (--region brazil / --region global).
- Validação temporal causal (persistência de 2/3 janelas).
- Exportação em CSV, JSON e visualização detalhada em linha do tempo.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp, softmax
import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import arduino_frontend
from arduino_mic_emulator import (
    TARGET_SR,
    emulate_arduino_mic,
    load_audio_via_ffmpeg,
)
from robust_audio_features import analyze_harmonic_flutter
from train_hierarchical_models import BRAZIL_PRIORS

MODELS_DIR = ROOT / "results/arduino_species/models"
PRESENCE_DIR = ROOT / "results/arduino/models"
WINDOW_SAMPLES = arduino_frontend.WINDOW_SAMPLES  # 15872 samples = 0.992 s @ 16 kHz


def download_google_drive_file(url_or_id: str, destination: Path) -> Path:
    """Baixa um arquivo público do Google Drive via URL ou ID."""
    match = re.search(r"[-_\w]{25,}", url_or_id)
    if not match:
        raise ValueError(f"Não foi possível extrair ID do Google Drive de: {url_or_id}")
    file_id = match.group(0)

    print(f"-> Baixando arquivo do Google Drive (ID: {file_id})...")
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

    print(f"-> Arquivo salvo em: {destination} ({destination.stat().st_size / 1024:.1f} KB)")
    return destination


class HierarchicalMosquitoPipeline:
    def __init__(self, model_mode: str = "deployed", region: str = "global"):
        self.model_mode = model_mode
        self.region = region.lower()
        self.models = self._load_models()
        self.classes = self.models[0]["classes"]
        self.num_classes = len(self.classes)
        self.genus_classes = np.array([c.split()[0] for c in self.classes])
        self.unique_genera = sorted(list(set(self.genus_classes)))

    def _load_models(self) -> list[dict]:
        """Carrega os pesos calibrados treinados com separação estrita por grupo/dobra."""
        loaded = []
        if self.model_mode == "deployed":
            fold_file = MODELS_DIR / "species_fold2.npz"
            loaded.append(self._parse_fold_file(fold_file))
        elif self.model_mode == "ensemble":
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
        classes = [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in data["classes"]]
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

    def predict_presence(self, feat68: np.ndarray) -> np.ndarray:
        """Calcula a probabilidade de presença de mosquito em [0, 1]."""
        feat68 = np.atleast_2d(feat68).astype("float32")
        scores = []
        for m in self.models:
            h = np.maximum(0, feat68 @ m["presence_w"][0].T + m["presence_b"][0])
            raw = h @ m["presence_out_w"] + m["presence_out_b"]
            z = m["presence_slope"] * raw + m["presence_intercept"]
            scores.append(expit(z))
        return np.mean(scores, axis=0)

    def predict_species_probabilities(self, feat68: np.ndarray) -> tuple[np.ndarray, float]:
        """Calcula as probabilidades das 20 espécies e o valor de energia livre."""
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

        mean_probs = np.mean(probs_list, axis=0)
        mean_energy = float(np.mean(energies))

        # Aplica prior biogeográfico regional se solicitado
        if self.region == "brazil":
            probs_adj = np.zeros_like(mean_probs)
            for i, sp in enumerate(self.classes):
                gen = sp.split()[0]
                prior = BRAZIL_PRIORS.get(gen, {}).get(sp, 0.05)
                probs_adj[:, i] = mean_probs[:, i] * prior
            probs_adj /= np.maximum(probs_adj.sum(axis=1, keepdims=True), 1e-12)
            mean_probs = probs_adj

        return mean_probs[0], mean_energy

    def evaluate_windows(
        self,
        features: np.ndarray,
        windows_pcm: list[np.ndarray],
        presence_threshold: float | None = None,
        confidence_threshold: float | None = None,
        energy_threshold: float = -20.0,
    ) -> list[dict]:
        """Avaliação hierárquica janela a janela com filtro de harmônicos de rede e persistência causal."""
        features = np.asarray(features, dtype="float32")
        n_windows = len(features)
        results = []

        if presence_threshold is None:
            presence_threshold = float(self.models[0]["presence_threshold"])
        if confidence_threshold is None:
            confidence_threshold = float(self.models[0]["confidence_threshold"])

        presence_probs = self.predict_presence(features)

        # Histórico causal para persistência de gênero (2 de 3 janelas)
        genus_history: list[str] = []
        species_history: list[str] = []

        for i in range(n_windows):
            feat = features[i : i + 1]
            p_pres = float(presence_probs[i])
            w_pcm = windows_pcm[i]

            # Análise harmônica e dinâmica aerodinâmica
            flutter_info = analyze_harmonic_flutter(w_pcm, sr=TARGET_SR)
            is_60hz = flutter_info["is_electrical_60hz"]
            f0_mean = flutter_info["f0_mean"]
            f0_std = flutter_info["f0_std"]
            h_ratio = flutter_info["harmonic_ratio"]

            # Detecção de Presença Tier 1 (Combina Evidência Biofísica de Voo e Rede Neural)
            is_bio_flight = bool(
                h_ratio >= 0.08
                and 300.0 <= f0_mean <= 850.0
                and 10.0 <= f0_std <= 85.0
                and not is_60hz
            )
            is_neural_detected = bool(p_pres >= presence_threshold and h_ratio >= 0.04 and not is_60hz)
            is_detected = is_bio_flight or is_neural_detected

            if is_60hz:
                status = f"RUÍDO ELÉTRICO/MECÂNICO ({f0_mean:.0f} Hz, harmônico 60Hz)"
            elif not is_detected:
                status = "AUSENTE (RUÍDO)"
            else:
                status = "CANDIDATO"

            if is_detected:
                sp_probs, free_energy = self.predict_species_probabilities(feat)
                is_ood = bool(free_energy > energy_threshold)

                # Tier 2: Probabilidades por Gênero
                genus_probs = {}
                for g in self.unique_genera:
                    mask = [gc == g for gc in self.genus_classes]
                    genus_probs[g] = float(sp_probs[mask].sum())

                top_genus = max(genus_probs, key=genus_probs.get)
                genus_conf = genus_probs[top_genus]

                # Tier 3: Espécie
                sp_idx = int(np.argmax(sp_probs))
                cand_species = self.classes[sp_idx]
                sp_conf = float(sp_probs[sp_idx])

                accepted = bool(sp_conf >= confidence_threshold and not is_ood)

                # Persistência causal
                genus_history.append(top_genus)
                genus_history = genus_history[-3:]
                species_history.append(cand_species if accepted else "---")
                species_history = species_history[-3:]

                confirmed_genus = None
                if len(genus_history) == 3 and genus_history.count(top_genus) >= 2:
                    confirmed_genus = top_genus

                confirmed_species = None
                if len(species_history) == 3 and cand_species != "---" and species_history.count(cand_species) >= 2:
                    confirmed_species = cand_species

                if is_ood:
                    status = "INCERTO (FORA DA DISTRIBUIÇÃO)"
                elif confirmed_genus and confirmed_species:
                    status = f"CONFIRMADO ({confirmed_species})"
                elif confirmed_genus:
                    status = f"GÊNERO CONFIRMADO ({confirmed_genus})"
                elif accepted:
                    status = f"CANDIDATO ({top_genus}: {cand_species})"
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
                    "detected": is_detected,
                    "f0_hz": f0_mean,
                    "f0_std": f0_std,
                    "harmonic_ratio": h_ratio,
                    "is_electrical_60hz": is_60hz,
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
                }
            )

        return results


def process_audio_file(
    audio_path: str | Path,
    emulate_mic: bool = False,
    hop_seconds: float = 0.5,
    classifier: HierarchicalMosquitoPipeline | None = None,
    presence_threshold: float | None = None,
    confidence_threshold: float | None = None,
    energy_threshold: float = -20.0,
) -> tuple[pd.DataFrame, dict]:
    """Processa um arquivo de áudio de ponta a ponta e retorna relatório hierárquico."""
    if classifier is None:
        classifier = HierarchicalMosquitoPipeline(model_mode="deployed")

    raw_audio, sr = load_audio_via_ffmpeg(audio_path, target_sr=TARGET_SR)
    duration = len(raw_audio) / float(TARGET_SR)

    if emulate_mic:
        processed_pcm = emulate_arduino_mic(raw_audio, sr=TARGET_SR, inject_noise=True, quantize=True).ravel()
    else:
        processed_pcm = np.asarray(arduino_frontend.pcm16(raw_audio), dtype=np.int16).ravel()

    hop_samples = int(round(hop_seconds * TARGET_SR))
    total_samples = len(processed_pcm)

    windows = []
    starts_s = []
    ends_s = []

    pos = 0
    while pos + WINDOW_SAMPLES <= total_samples:
        window = processed_pcm[pos : pos + WINDOW_SAMPLES]
        windows.append(window)
        starts_s.append(pos / float(TARGET_SR))
        ends_s.append((pos + WINDOW_SAMPLES) / float(TARGET_SR))
        pos += hop_samples

    if len(windows) == 0:
        raise ValueError(f"Áudio com duração ({duration:.2f} s) menor que a janela mínima (0.992 s).")

    windows_arr = np.stack(windows)
    feats = arduino_frontend.features(windows_arr, replay_pcm=False)

    eval_rows = classifier.evaluate_windows(
        feats,
        windows_pcm=windows,
        presence_threshold=presence_threshold,
        confidence_threshold=confidence_threshold,
        energy_threshold=energy_threshold,
    )

    for i, row in enumerate(eval_rows):
        row["start_s"] = round(starts_s[i], 3)
        row["end_s"] = round(ends_s[i], 3)

    df = pd.DataFrame(eval_rows)

    # Agregação do Diagnóstico Geral
    detected_mask = df.detected.to_numpy()
    total_windows = len(df)
    detected_windows = int(np.sum(detected_mask))
    active_ratio = detected_windows / float(total_windows)

    pos_df = df[df.detected]
    genus_counts = pos_df.genus_candidate.value_counts().to_dict() if len(pos_df) else {}
    species_counts = pos_df.candidate_species.value_counts().to_dict() if len(pos_df) else {}
    confirmed_genus_counts = (
        df.confirmed_genus.dropna().value_counts().to_dict() if df.confirmed_genus.notna().any() else {}
    )
    confirmed_sp_counts = (
        df.confirmed_species.dropna().value_counts().to_dict() if df.confirmed_species.notna().any() else {}
    )

    top_genus = max(genus_counts, key=genus_counts.get) if genus_counts else None
    top_confirmed_genus = max(confirmed_genus_counts, key=confirmed_genus_counts.get) if confirmed_genus_counts else None
    top_species = max(species_counts, key=species_counts.get) if species_counts else None
    top_confirmed_sp = max(confirmed_sp_counts, key=confirmed_sp_counts.get) if confirmed_sp_counts else None

    # Média de F0 nas janelas de voo real
    flight_f0 = float(pos_df.f0_hz.median()) if len(pos_df) else 0.0

    summary = {
        "audio_file": str(audio_path),
        "duration_seconds": round(duration, 2),
        "emulate_arduino_mic": emulate_mic,
        "region_prior": classifier.region,
        "total_windows": total_windows,
        "detected_windows": detected_windows,
        "active_ratio_pct": round(active_ratio * 100, 2),
        "mosquito_present": bool(active_ratio >= 0.10 or detected_windows >= 3),
        "median_flight_f0_hz": round(flight_f0, 1),
        "primary_genus": top_genus,
        "confirmed_genus": top_confirmed_genus,
        "primary_species": top_species,
        "confirmed_species": top_confirmed_sp,
        "genus_distribution": genus_counts,
        "species_distribution": species_counts,
    }

    return df, summary


def print_cli_report(summary: dict, df: pd.DataFrame):
    """Exibe relatório estruturado e linha do tempo no terminal."""
    print("\n" + "=" * 75)
    print("        RELATÓRIO HIERÁRQUICO DE IDENTIFICAÇÃO BIOACÚSTICA (ML)         ")
    print("=" * 75)
    print(f"Arquivo:                  {summary['audio_file']}")
    print(f"Duração Total:            {summary['duration_seconds']} s")
    print(f"Emulação Mic Arduino:     {'ATIVADA (ST MP34DT05)' if summary['emulate_arduino_mic'] else 'DESATIVADA (Áudio Bruto)'}")
    print(f"Prior Epidemiológico:     {summary['region_prior'].upper()}")
    print(f"Total de Janelas:         {summary['total_windows']} ({summary['detected_windows']} de voo detectado)")
    print(f"Atividade de Voo:         {summary['active_ratio_pct']}% da gravação")
    print(f"Diagnóstico Tier 1:       {'MOSQUITO DETECTADO' if summary['mosquito_present'] else 'NENHUM MOSQUITO IDENTIFICADO'}")

    if summary["mosquito_present"]:
        print(f"Frequência Fundamental:   F0 Mediana = {summary['median_flight_f0_hz']:.1f} Hz")
        print("\n--- TIER 2: VETOR EPIDEMIOLÓGICO (GÊNERO - ALTA CERTEZA >93%) ---")
        gen_str = summary["confirmed_genus"] or summary["primary_genus"]
        print(f"Gênero Identificado:      {gen_str} (Confirmação Causal 2/3: {summary['confirmed_genus'] is not None})")
        print("Distribuição por Gênero:")
        for g, count in summary["genus_distribution"].items():
            pct = count / summary["detected_windows"] * 100
            print(f"  * {g:<14}: {count:2d} janelas ({pct:5.1f}%)")

        print("\n--- TIER 3: ESPÉCIE ESPECÍFICA (COM SUPORTE A PRIORS) ---")
        sp_str = summary["confirmed_species"] or summary["primary_species"]
        print(f"Espécie Candidata:        {sp_str}")
        print("Top Espécies Candidatas:")
        for sp, count in list(summary["species_distribution"].items())[:5]:
            pct = count / summary["detected_windows"] * 100
            print(f"  * {sp:<26}: {count:2d} janelas ({pct:5.1f}%)")

    print("\n--- LINHA DO TEMPO DETALHADA ---")
    active_subset = df[df.detected].head(15) if summary["mosquito_present"] else df.head(10)
    print(f"{'Tempo (s)':<12} | {'F0 (Hz)':<7} | {'Gênero':<12} | {'Espécie':<24} | {'Status'}")
    print("-" * 75)
    for _, r in active_subset.iterrows():
        t_str = f"{r['start_s']:4.1f}-{r['end_s']:4.1f}s"
        gen_str = str(r["genus_candidate"] or "---")[:12]
        sp_str = str(r["candidate_species"] or "---")[:24]
        f0_val = f"{r['f0_hz']:5.1f}" if r["f0_hz"] > 0 else " --- "
        print(f"{t_str:<12} | {f0_val:<7} | {gen_str:<12} | {sp_str:<24} | {r['status']}")

    if len(df[df.detected]) > 15:
        print(f"... e mais {len(df[df.detected]) - 15} janelas ativas.")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnóstico Hierárquico de Presença, Gênero e Espécie de Mosquito via ML."
    )
    parser.add_argument("audio", type=str, help="Caminho do áudio local (.wav, .mp3) ou link/ID do Google Drive")
    parser.add_argument("--emulate-arduino", action="store_true", help="Aplica emulação acústica do mic MP34DT05")
    parser.add_argument("--compare", action="store_true", help="Executa comparação lado a lado (Bruto vs Mic Arduino)")
    parser.add_argument("--hop", type=float, default=0.5, help="Passo temporal em segundos (padrão: 0.5s)")
    parser.add_argument("--model", choices=["deployed", "ensemble"], default="ensemble", help="Modo do modelo (padrão: ensemble)")
    parser.add_argument("--region", choices=["brazil", "global"], default="brazil", help="Prior biogeográfico regional (padrão: brazil)")
    parser.add_argument("--presence-threshold", type=float, default=None, help="Limiar de presença")
    parser.add_argument("--confidence-threshold", type=float, default=None, help="Limiar de confiança")
    parser.add_argument("--output-csv", type=str, default=None, help="Caminho para salvar o CSV detalhado")
    parser.add_argument("--output-json", type=str, default=None, help="Caminho para salvar o resumo em JSON")

    args = parser.parse_args()

    audio_arg = args.audio
    temp_download = None

    if "drive.google.com" in audio_arg or len(audio_arg) > 25 and "/" not in audio_arg and not Path(audio_arg).exists():
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
            print("\n[MODO DE COMPARAÇÃO ATIVADO: ÁUDIO BRUTO vs MICROFONE ARDUINO NANO 33 BLE SENSE]")
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

            print("\n" + "=" * 75)
            print("                 RESUMO COMPARATIVO DE DOMÍNIO ACÚSTICO                  ")
            print("=" * 75)
            print(f"{'Métrica':<30} | {'Áudio Bruto':<18} | {'Mic Arduino (MP34DT05)':<18}")
            print("-" * 75)
            print(f"{'Presença Detectada':<30} | {str(sum_raw['mosquito_present']):<18} | {str(sum_emu['mosquito_present']):<18}")
            print(f"{'Atividade (% do tempo)':<30} | {sum_raw['active_ratio_pct']}%{'':<12} | {sum_emu['active_ratio_pct']}%{'':<12}")
            print(f"{'Janelas de Voo':<30} | {sum_raw['detected_windows']}/{sum_raw['total_windows']}{'':<10} | {sum_emu['detected_windows']}/{sum_emu['total_windows']}{'':<10}")
            print(f"{'Gênero Confirmado':<30} | {str(sum_raw['confirmed_genus']):<18} | {str(sum_emu['confirmed_genus']):<18}")
            print(f"{'Espécie Candidata':<30} | {str(sum_raw['primary_species']):<18} | {str(sum_emu['primary_species']):<18}")
            print("=" * 75 + "\n")

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
