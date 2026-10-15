"""Audio file decoding helpers used by prediction scripts."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav


def load_audio_via_ffmpeg(audio_path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Decode common audio formats to mono float audio at the requested rate."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_path}")
    try:
        import soundfile as sf
        data, sr = sf.read(str(audio_path), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != target_sr:
            gcd = np.gcd(sr, target_sr)
            data = signal.resample_poly(data, target_sr // gcd, sr // gcd)
            sr = target_sr
        return data.astype(np.float32), sr
    except Exception:
        pass
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_wav_path = tmp_file.name
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path), "-ar", str(target_sr), "-ac", "1",
             "-c:a", "pcm_s16le", tmp_wav_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        if result.returncode:
            raise RuntimeError(f"Erro ao converter áudio com ffmpeg: {result.stderr.decode(errors='ignore')}")
        sr, data = wav.read(tmp_wav_path)
        data = data.astype(np.float32) / 32768.0
        return data, sr
    finally:
        if os.path.exists(tmp_wav_path):
            os.remove(tmp_wav_path)
