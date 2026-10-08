"""Emulador Acústico do Microfone Arduino Nano 33 BLE Sense (ST MP34DT05).

Modela com fidelidade a cadeia eletroacústica do microfone MEMS PDM MP34DT05:
1. Resampling para 16.000 Hz (frequência de amostragem padrão da biblioteca PDM).
2. Resposta em frequência de hardware:
   - Filtro passa-altas de 4ª ordem @ 100 Hz (rejeição de offset DC e ruído mecânico).
   - Filtro passa-baixas de 6ª ordem @ 7.200 Hz (filtro de decimação PDM anti-aliasing).
   - Filtro de pico (peaking IIR) @ 6.500 Hz (Q=2.5, ganho +4.5 dB) modelando a ressonância
     da cavidade acústica do encapsulamento MEMS conforme o datasheet da STMicroelectronics.
3. Injeção de piso de ruído térmico/PDM:
   - SNR especificado de 64 dB(A), simulando ruído elétrico/térmico a -64 dBFS.
4. Quantização digital PCM 16 bits:
   - Simulação da resolução do conversor e arredondamento inteiro int16.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav

TARGET_SR = 16000
MIC_SNR_DB = 64.0  # ST MP34DT05 datasheet: 64 dB SNR (A-weighted)
RESONANCE_FREQ = 6500.0  # Frequência de pico de ressonância do encapsulamento MEMS
RESONANCE_Q = 2.5
RESONANCE_GAIN_DB = 4.5
HPF_CUTOFF = 100.0
LPF_CUTOFF = 7200.0


def _build_filters(fs: int = TARGET_SR):
    """Projeta os coeficientes IIR do circuito de transdução MEMS."""
    # 1. Filtro Passa-Altas (Butterworth 4ª ordem)
    sos_hp = signal.butter(4, HPF_CUTOFF / (fs / 2.0), btype="highpass", output="sos")

    # 2. Filtro Passa-Baixas (Butterworth 6ª ordem anti-aliasing de decimação PDM)
    sos_lp = signal.butter(6, LPF_CUTOFF / (fs / 2.0), btype="lowpass", output="sos")

    # 3. Peaking EQ em 6.5 kHz (Ressonância da cavidade MEMS)
    w0 = 2.0 * np.pi * RESONANCE_FREQ / fs
    A = 10.0 ** (RESONANCE_GAIN_DB / 40.0)
    alpha = np.sin(w0) / (2.0 * RESONANCE_Q)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / A
    sos_peak = np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)

    return sos_hp, sos_lp, sos_peak


def load_audio_via_ffmpeg(audio_path: str | Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Decodifica qualquer formato de áudio (wav, mp3, m4a, ogg, flac) via ffmpeg ou soundfile."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_path}")

    # Tentativa direta com soundfile (evita dependência de ffmpeg para WAV/FLAC/OGG)
    try:
        import soundfile as sf
        data, sr = sf.read(str(audio_path), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != target_sr:
            gcd = np.gcd(sr, target_sr)
            up = target_sr // gcd
            down = sr // gcd
            data = signal.resample_poly(data, up, down)
            sr = target_sr
        return data.astype(np.float32), sr
    except Exception:
        pass

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_wav_path = tmp_file.name

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(audio_path),
            "-ar", str(target_sr),
            "-ac", "1",
            "-c:a", "pcm_s16le",
            tmp_wav_path,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Erro ao converter áudio com ffmpeg: {result.stderr.decode('utf-8', errors='ignore')}")

        sr, data = wav.read(tmp_wav_path)
        data = data.astype(np.float32)
        # Normalização para [-1.0, 1.0] caso estivesse em int16
        if data.dtype != np.float32:
            data = data / 32768.0
        else:
            data = data / 32768.0
        return data, sr
    finally:
        if os.path.exists(tmp_wav_path):
            os.remove(tmp_wav_path)


def emulate_arduino_mic(
    audio: np.ndarray,
    sr: int = TARGET_SR,
    inject_noise: bool = True,
    quantize: bool = True,
    seed: int | None = 42,
) -> np.ndarray:
    """Aplica a degradação e resposta acústica do microfone ST MP34DT05.

    Parâmetros:
        audio: Array 1D com amostras de áudio float32 normalizadas no intervalo [-1, 1].
        sr: Frequência de amostragem de entrada (se != 16000, será resamostrado).
        inject_noise: Se True, injeta ruído térmico/PDM a -64 dBFS SNR.
        quantize: Se True, quantiza em PCM 16-bit inteiro [-32768, 32767].
        seed: Semente opcional para reprodutibilidade do ruído.

    Retorna:
        Array com o áudio processado. Se quantize=True, retorna int16. Caso contrário, float32.
    """
    audio = np.asarray(audio, dtype=np.float32).ravel()

    # 1. Resampling se necessário
    if sr != TARGET_SR:
        num_target = int(round(len(audio) * float(TARGET_SR) / float(sr)))
        audio = signal.resample(audio, num_target).astype(np.float32)

    # 2. Filtragem acústica do microfone MEMS
    sos_hp, sos_lp, sos_peak = _build_filters(TARGET_SR)
    filtered = signal.sosfilt(sos_hp, audio)
    filtered = signal.sosfilt(sos_lp, filtered)
    filtered = signal.sosfilt(sos_peak, filtered)

    # 3. Piso de ruído do microfone MEMS (64 dB SNR)
    if inject_noise:
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        # Nível RMS de referência (se sinal tiver RMS quase zero, usa 0.1 como referência)
        signal_rms = float(np.sqrt(np.mean(filtered**2)))
        ref_rms = max(signal_rms, 0.05)
        noise_rms = ref_rms * (10.0 ** (-MIC_SNR_DB / 20.0))

        # Ruído branco com filtragem passa-altas para simular ruído de clock/térmico
        raw_noise = rng.normal(0.0, noise_rms, size=len(filtered)).astype(np.float32)
        filtered = filtered + raw_noise

    # 4. Controle de Ganho Automático e Quantização PCM 16 bits
    peak = np.max(np.abs(filtered))
    if peak > 0:
        # Manter headroom de ~0.9 para evitar clipping artificial no PDM
        scale_factor = 0.90 / max(peak, 0.90)
        filtered = filtered * scale_factor

    if quantize:
        int_pcm = np.clip(np.rint(filtered * 32767.0), -32768, 32767).astype(np.int16)
        return int_pcm

    return filtered.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Emulador de Áudio do Microfone Arduino Nano 33 BLE Sense (ST MP34DT05)"
    )
    parser.add_argument("input_audio", type=str, help="Caminho do arquivo de áudio de entrada")
    parser.add_argument(
        "output_audio",
        type=str,
        nargs="?",
        default=None,
        help="Caminho do arquivo WAV de saída (padrão: <nome>_arduino_mic.wav)",
    )
    parser.add_argument(
        "--no-noise", action="store_true", help="Desativa a injeção do piso de ruído térmico -64 dB"
    )
    parser.add_argument(
        "--float-output", action="store_true", help="Salva como float32 em vez de PCM int16"
    )

    args = parser.parse_args()

    input_path = Path(args.input_audio)
    if not input_path.exists():
        print(f"Erro: Arquivo {input_path} não encontrado.", file=sys.stderr)
        sys.exit(1)

    if args.output_audio is None:
        out_path = input_path.parent / f"{input_path.stem}_arduino_mic.wav"
    else:
        out_path = Path(args.output_audio)

    print(f"-> Lendo e decodificando áudio: {input_path}")
    raw_audio, sr = load_audio_via_ffmpeg(input_path)
    dur = len(raw_audio) / float(sr)
    print(f"   Duração: {dur:.2f} s | Freq original: {sr} Hz")

    print(f"-> Aplicando perfil eletroacústico do sensor ST MP34DT05...")
    processed = emulate_arduino_mic(
        raw_audio,
        sr=sr,
        inject_noise=not args.no_noise,
        quantize=not args.float_output,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.float_output:
        wav.write(str(out_path), TARGET_SR, processed.astype(np.float32))
    else:
        wav.write(str(out_path), TARGET_SR, processed.astype(np.int16))

    print(f"-> Áudio emulado com sucesso gravado em: {out_path}")


if __name__ == "__main__":
    main()
