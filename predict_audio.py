"""Diagnóstico e Classificação de Áudio de Mosquitos com ML (Zero Data Leakage).

Executa detecção de presença de mosquitos e classificação de espécies (20 espécies)
a partir de qualquer arquivo de áudio ou link do Google Drive.

Suporta:
- Emulação da resposta acústica do microfone ST MP34DT05 (Arduino Nano 33 BLE Sense).
- Comparação lado a lado (áudio bruto vs áudio emulado).
- Extração em streaming de 68 características FFT (idêntica ao firmware Arduino).
- Detetor neural de presença calibrado com controle de falsos positivos.
- Classificador de espécies com escala de temperatura e rejeição OOD via Helmholtz Free Energy (Liu et al., NeurIPS 2020).
- Votação causal de persistência temporal (2/3 janelas contíguas).
- Exportação em CSV, JSON e visualização em linha do tempo.
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

    # Lida com telas de confirmação de arquivos grandes
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


class MosquitoClassifierPipeline:
    def __init__(self, model_mode: str = "deployed"):
        self.model_mode = model_mode
        self.models = self._load_models()
        self.classes = self.models[0]["classes"]
        self.num_classes = len(self.classes)

    def _load_models(self) -> list[dict]:
        """Carrega os pesos calibrados treinados com separação estrita por grupo/dobra."""
        loaded = []
        if self.model_mode == "deployed":
            # Modelo da Dobra 2: arquitetura exata embarcada no firmware do Arduino Nano 33 BLE Sense
            fold_file = MODELS_DIR / "species_fold2.npz"
            loaded.append(self._parse_fold_file(fold_file))
        elif self.model_mode == "ensemble":
            # Ensemble das 3 dobras (0, 1, 2) para máxima robustez em gravações de campo
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
            # Camada oculta 16 neurônios ReLU
            h = np.maximum(0, feat68 @ m["presence_w"][0].T + m["presence_b"][0])
            raw = h @ m["presence_out_w"] + m["presence_out_b"]
            z = m["presence_slope"] * raw + m["presence_intercept"]
            p = expit(z)
            scores.append(p)
        return np.mean(scores, axis=0)

    def predict_species_logits(self, feat68: np.ndarray) -> tuple[np.ndarray, float]:
        """Calcula os logits brutos das 20 espécies e temperatura."""
        feat68 = np.atleast_2d(feat68).astype("float32")
        logits_list = []
        temps = []
        for m in self.models:
            x = feat68
            for i, (w, b) in enumerate(zip(m["weights"], m["biases"])):
                x = x @ w.T + b
                if i < len(m["weights"]) - 1:
                    x = np.maximum(0, x)
            logits_list.append(x)
            temps.append(m["temperature"])
        mean_logits = np.mean(logits_list, axis=0)
        mean_temp = float(np.mean(temps))
        return mean_logits, mean_temp

    def evaluate_windows(
        self,
        features: np.ndarray,
        presence_threshold: float | None = None,
        confidence_threshold: float | None = None,
        energy_threshold: float = -20.0,
    ) -> list[dict]:
        """Avalia janelas consecutivas com detecção, classificação e energia de abstinência."""
        features = np.asarray(features, dtype="float32")
        n_windows = len(features)
        results = []

        if presence_threshold is None:
            presence_threshold = float(self.models[0]["presence_threshold"])
        if confidence_threshold is None:
            confidence_threshold = float(self.models[0]["confidence_threshold"])

        presence_probs = self.predict_presence(features)

        # Histórico para confirmação causal de persistência temporal (2 de 3)
        history: list[int] = []

        for i in range(n_windows):
            feat = features[i : i + 1]
            p_pres = float(presence_probs[i])
            is_detected = bool(p_pres >= presence_threshold)

            if is_detected:
                logits, T = self.predict_species_logits(feat)
                probs = softmax(logits / T, axis=1)[0]
                # Pontuação de Energia Livre de Helmholtz (Liu et al., NeurIPS 2020)
                # E(x) = -T * logsumexp(z / T)
                # Janelas com ruído anômalo ou inseto desconhecido produzem energia elevada (próxima ou superior ao limiar)
                free_energy = float(-T * logsumexp(logits / T, axis=1)[0])

                pred_idx = int(np.argmax(probs))
                pred_conf = float(probs[pred_idx])
                cand_species = self.classes[pred_idx]

                is_ood = bool(free_energy > energy_threshold)
                accepted = bool(pred_conf >= confidence_threshold and not is_ood)

                # Atualiza histórico causal
                history.append(pred_idx if accepted else -1)
                history = history[-3:]

                # Confirmação temporal: pelo menos 2 janelas nas últimas 3 com o mesmo candidato aceito
                confirmed_idx = -1
                if len(history) == 3 and pred_idx >= 0 and accepted:
                    if history.count(pred_idx) >= 2:
                        confirmed_idx = pred_idx

                confirmed_species = self.classes[confirmed_idx] if confirmed_idx >= 0 else None
                status = "CONFIRMADO" if confirmed_species else ("INCERTO (BAIXA CONFIANÇA)" if not accepted else "CANDIDATO")
                if is_ood:
                    status = "INCERTO (FORA DA DISTRIBUIÇÃO / RUÍDO)"
            else:
                probs = np.zeros(self.num_classes, dtype="float32")
                cand_species = None
                pred_conf = 0.0
                free_energy = 0.0
                confirmed_species = None
                status = "AUSENTE (RUÍDO)"
                history.append(-1)
                history = history[-3:]

            results.append(
                {
                    "window_index": i,
                    "presence_score": p_pres,
                    "detected": is_detected,
                    "candidate_species": cand_species,
                    "confidence": pred_conf,
                    "free_energy": free_energy,
                    "confirmed_species": confirmed_species,
                    "status": status,
                    "probabilities": probs.tolist(),
                }
            )

        return results


def process_audio_file(
    audio_path: str | Path,
    emulate_mic: bool = False,
    hop_seconds: float = 0.5,
    classifier: MosquitoClassifierPipeline | None = None,
    presence_threshold: float | None = None,
    confidence_threshold: float | None = None,
    energy_threshold: float = -20.0,
) -> tuple[pd.DataFrame, dict]:
    """Processa um arquivo de áudio de ponta a ponta e retorna relatório analítico."""
    if classifier is None:
        classifier = MosquitoClassifierPipeline(model_mode="deployed")

    # 1. Carrega áudio decodificado em 16 kHz
    raw_audio, sr = load_audio_via_ffmpeg(audio_path, target_sr=TARGET_SR)
    duration = len(raw_audio) / float(TARGET_SR)

    # 2. Aplica emulação do microfone Arduino se solicitado
    if emulate_mic:
        processed_pcm = emulate_arduino_mic(raw_audio, sr=TARGET_SR, inject_noise=True, quantize=True).ravel()
    else:
        # Se raw, quantiza para PCM 16-bit com ganho controlado
        processed_pcm = np.asarray(arduino_frontend.pcm16(raw_audio), dtype=np.int16).ravel()

    # 3. Fatiamento em janelas
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

    # 4. Extração de características (68 FFT bands + flatness + peak ratios)
    feats = arduino_frontend.features(windows_arr, replay_pcm=False)

    # 5. Avaliação com ML
    eval_rows = classifier.evaluate_windows(
        feats,
        presence_threshold=presence_threshold,
        confidence_threshold=confidence_threshold,
        energy_threshold=energy_threshold,
    )

    for i, row in enumerate(eval_rows):
        row["start_s"] = round(starts_s[i], 3)
        row["end_s"] = round(ends_s[i], 3)

    df = pd.DataFrame(eval_rows)

    # 6. Agregação do Diagnóstico Geral do Arquivo
    detected_mask = df.detected.to_numpy()
    total_windows = len(df)
    detected_windows = int(np.sum(detected_mask))
    active_ratio = detected_windows / float(total_windows)

    pos_df = df[df.detected]
    species_counts = pos_df.candidate_species.value_counts().to_dict() if len(pos_df) else {}
    confirmed_counts = (
        df.confirmed_species.dropna().value_counts().to_dict() if df.confirmed_species.notna().any() else {}
    )

    top_candidate = max(species_counts, key=species_counts.get) if species_counts else None
    top_confirmed = max(confirmed_counts, key=confirmed_counts.get) if confirmed_counts else None

    mean_pres = float(df.presence_score.mean())
    max_pres = float(df.presence_score.max())

    summary = {
        "audio_file": str(audio_path),
        "duration_seconds": round(duration, 2),
        "emulate_arduino_mic": emulate_mic,
        "total_windows": total_windows,
        "detected_windows": detected_windows,
        "active_ratio_pct": round(active_ratio * 100, 2),
        "mosquito_present": bool(active_ratio >= 0.15 or detected_windows >= 3),
        "mean_presence_score": round(mean_pres, 4),
        "max_presence_score": round(max_pres, 4),
        "primary_candidate_species": top_candidate,
        "primary_confirmed_species": top_confirmed,
        "candidate_distribution": species_counts,
        "confirmed_distribution": confirmed_counts,
    }

    return df, summary


def print_cli_report(summary: dict, df: pd.DataFrame):
    """Exibe relatório estruturado e linha do tempo no terminal."""
    print("\n" + "=" * 70)
    print("           RELATÓRIO DE IDENTIFICAÇÃO BIOACÚSTICA (ML)           ")
    print("=" * 70)
    print(f"Arquivo:                  {summary['audio_file']}")
    print(f"Duração Total:            {summary['duration_seconds']} s")
    print(f"Emulação Mic Arduino:     {'ATIVADA (ST MP34DT05)' if summary['emulate_arduino_mic'] else 'DESATIVADA (Áudio Bruto)'}")
    print(f"Total de Janelas:         {summary['total_windows']} ({summary['detected_windows']} com mosquito detectado)")
    print(f"Atividade de Voo:         {summary['active_ratio_pct']}% da gravação")
    print(f"Diagnóstico de Presença:  {'MOSQUITO DETECTADO' if summary['mosquito_present'] else 'NENHUM MOSQUITO IDENTIFICADO'}")
    print(f"Score Médio de Presença:  {summary['mean_presence_score']:.3f} (Pico: {summary['max_presence_score']:.3f})")

    if summary["mosquito_present"]:
        print("\n--- CLASSIFICAÇÃO TAXONÔMICA ---")
        if summary["primary_confirmed_species"]:
            print(f"Espécie Confirmada (Voto 2/3):  {summary['primary_confirmed_species']}")
        else:
            print(f"Espécie Candidata Predominante: {summary['primary_candidate_species']} (Aguardando persistência causal)")

        print("\nDistribuição de Janelas por Espécie:")
        for sp, count in summary["candidate_distribution"].items():
            pct = count / summary["detected_windows"] * 100
            print(f"  - {sp:<26}: {count:2d} janelas ({pct:5.1f}%)")

    print("\n--- LINHA DO TEMPO (Resumo das Primeiras / Ativas Janelas) ---")
    active_subset = df[df.detected].head(15) if summary["mosquito_present"] else df.head(10)
    print(f"{'Tempo (s)':<14} | {'Presença':<9} | {'Candidato':<26} | {'Conf':<6} | {'Status'}")
    print("-" * 75)
    for _, r in active_subset.iterrows():
        t_str = f"{r['start_s']:4.1f}-{r['end_s']:4.1f}s"
        sp_str = str(r["candidate_species"] or "---")[:26]
        print(f"{t_str:<14} | {r['presence_score']:<9.3f} | {sp_str:<26} | {r['confidence']:<6.2f} | {r['status']}")

    if len(df[df.detected]) > 15:
        print(f"... e mais {len(df[df.detected]) - 15} janelas ativas.")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnóstico de Presença e Espécie de Mosquito via ML (TinyML / Arduino)."
    )
    parser.add_argument("audio", type=str, help="Caminho do áudio local (.wav, .mp3, etc.) ou link/ID do Google Drive")
    parser.add_argument(
        "--emulate-arduino", action="store_true", help="Aplica o filtro eletroacústico do microfone MEMS MP34DT05"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Executa comparação lado a lado (Bruto vs Emulado no Mic Arduino)"
    )
    parser.add_argument(
        "--hop", type=float, default=0.5, help="Passo temporal entre janelas consecutivas em segundos (padrão: 0.5s)"
    )
    parser.add_argument(
        "--model", choices=["deployed", "ensemble"], default="deployed", help="Modo do modelo: deployed (Dobra 2 firmware) ou ensemble"
    )
    parser.add_argument(
        "--presence-threshold", type=float, default=None, help="Limiar de presença (padrão: calibrado pelo modelo)"
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=None, help="Limiar de confiança da espécie (padrão: calibrado 0.48)"
    )
    parser.add_argument(
        "--output-csv", type=str, default=None, help="Caminho para salvar o CSV detalhado janela a janela"
    )
    parser.add_argument(
        "--output-json", type=str, default=None, help="Caminho para salvar o resumo em JSON"
    )

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
        classifier = MosquitoClassifierPipeline(model_mode=args.model)

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

            print("\n" + "=" * 70)
            print("                 RESUMO COMPARATIVO DE DOMÍNIO ACÚSTICO                  ")
            print("=" * 70)
            print(f"{'Métrica':<30} | {'Áudio Bruto':<18} | {'Mic Arduino (MP34DT05)':<18}")
            print("-" * 70)
            print(f"{'Presença Detectada':<30} | {str(sum_raw['mosquito_present']):<18} | {str(sum_emu['mosquito_present']):<18}")
            print(f"{'Atividade (% do tempo)':<30} | {sum_raw['active_ratio_pct']}%{'':<12} | {sum_emu['active_ratio_pct']}%{'':<12}")
            print(f"{'Janelas Ativas':<30} | {sum_raw['detected_windows']}/{sum_raw['total_windows']}{'':<10} | {sum_emu['detected_windows']}/{sum_emu['total_windows']}{'':<10}")
            print(f"{'Candidato Principal':<30} | {str(sum_raw['primary_candidate_species']):<18} | {str(sum_emu['primary_candidate_species']):<18}")
            print(f"{'Confirmado Causal (2/3)':<30} | {str(sum_raw['primary_confirmed_species']):<18} | {str(sum_emu['primary_confirmed_species']):<18}")
            print("=" * 70 + "\n")

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
