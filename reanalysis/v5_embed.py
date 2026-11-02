"""v5 (exploratória): embeddings congelados de encoders pré-treinados em áudio massivo (H2).

Mesma entrada para todos: cache acústico de 8 kHz (v5_data), normalização pelo piso de ruído da própria
gravação (v3_embed.whiten), até 6 trechos de 5 s espaçados (trecho curto completado com zeros no centro),
ganho RMS 0,05 por trecho, reamostragem para a taxa do modelo e média dos embeddings por arquivo.

  perch       Perch 2.0 (ONNX), só para as linhas sem cache da v3 (Funchal)
  ast         MIT/ast-finetuned-audioset-10-10-0.4593 (AudioSet), pooler 768 + logits de insetos do AudioSet
  clap        laion/clap-htsat-unfused (LAION-Audio-630K), 512
  biolingual  davidrrobinson/BioLingual (CLAP em sons de animais), 512 + similaridade com nomes de espécies

Uso: python -m reanalysis.v5_embed {perch,ast,clap,biolingual}
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from reanalysis.v3_embed import whiten

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache/v5"
CHUNK_8K = 40000  # 5 s a 8 kHz, como os 160.000 a 32 kHz da v3
MAX_CHUNKS = 6
SPECIES_TEXT = ["Aedes aegypti", "Aedes albopictus", "Culex quinquefasciatus", "Culex pipiens", "Anopheles gambiae",
                "Anopheles arabiensis", "Anopheles stephensi", "Anopheles dirus", "Culiseta", "Aedes sierrensis"]
AST_LABELS = ["Mosquito", "Insect", "Fly, housefly", "Bee, wasp, etc.", "Buzz"]


def chunks(x):
    x = whiten(x)
    if len(x) <= CHUNK_8K:
        pad = CHUNK_8K - len(x)
        x = np.pad(x, (pad // 2, pad - pad // 2))
        starts = [0]
    else:
        n = len(x) // CHUNK_8K
        starts = np.unique(np.linspace(0, n - 1, min(n, MAX_CHUNKS)).round().astype(int)) * CHUNK_8K
    out = []
    for s in starts:
        c = x[s:s + CHUNK_8K].astype(np.float64)
        out.append((c / max(np.sqrt(np.mean(c ** 2)), 1e-8) * 0.05).astype(np.float32))
    return out


def iterate(meta, waves, rate, rows=None):
    up = rate // 8000
    for i in (range(len(meta)) if rows is None else rows):
        r = meta.iloc[i]
        x = waves[r.offset:r.offset + r.length].astype(np.float32)
        yield i, [resample_poly(c, up, 1).astype(np.float32) for c in chunks(x)]


def run(name, batch=24):
    import torch
    meta = pd.read_csv(CACHE / "acoustic_meta.csv")
    waves = np.load(CACHE / "acoustic_waves.npy", mmap_mode="r")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extra = {}
    if name == "ast":
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        rate, repo = 16000, "MIT/ast-finetuned-audioset-10-10-0.4593"
        fx = ASTFeatureExtractor.from_pretrained(repo)
        model = ASTForAudioClassification.from_pretrained(repo).to(device).eval()
        label_ids = [model.config.label2id[k] for k in AST_LABELS]

        def encode(clips):
            inputs = fx(clips, sampling_rate=rate, return_tensors="pt").to(device)
            out = model.audio_spectrogram_transformer(**inputs)
            pooled = out.pooler_output
            logits = model.classifier(pooled)[:, label_ids]
            return torch.cat([pooled, logits], 1)
    else:
        from transformers import ClapModel, ClapProcessor
        rate = 48000
        repo = "laion/clap-htsat-unfused" if name == "clap" else "davidrrobinson/BioLingual"
        proc = ClapProcessor.from_pretrained(repo)
        model = ClapModel.from_pretrained(repo).to(device).eval()
        with torch.no_grad():
            t = proc(text=[f"the sound of a {s} mosquito" for s in SPECIES_TEXT], return_tensors="pt", padding=True).to(device)
            text = torch.nn.functional.normalize(model.get_text_features(**t), dim=-1)
        extra["text_species"] = SPECIES_TEXT

        def encode(clips):
            inputs = proc(audio=clips, sampling_rate=rate, return_tensors="pt").to(device)
            a = model.get_audio_features(**inputs)
            sims = torch.nn.functional.normalize(a, dim=-1) @ text.T
            return torch.cat([a, sims], 1)

    vectors = [None] * len(meta)
    pending, owners = [], []

    def flush():
        with torch.no_grad():
            e = encode(pending).float().cpu().numpy()
        for owner, v in zip(owners, e):
            vectors[owner] = v if vectors[owner] is None else vectors[owner] + v
        pending.clear(), owners.clear()

    counts = np.zeros(len(meta))
    for i, clips in iterate(meta, waves, rate):
        for c in clips:
            pending.append(c), owners.append(i)
            counts[i] += 1
            if len(pending) >= batch:
                flush()
        if i % 1000 == 0:
            print(name, i, len(meta), flush=True)
    if pending:
        flush()
    E = np.stack(vectors) / counts[:, None]
    np.savez_compressed(CACHE / f"emb_{name}.npz", embedding=E.astype(np.float32), n_chunks=counts, **extra)
    print("saved", name, E.shape)


def perch_funchal():
    """Perch 2.0 só nas linhas sem cache da v3 (Funchal), com as funções da v3."""
    import onnxruntime as ort
    from reanalysis.v3_features import MODEL, audio_chunks
    meta = pd.read_csv(CACHE / "acoustic_meta.csv")
    waves = np.load(CACHE / "acoustic_waves.npy", mmap_mode="r")
    rows = np.flatnonzero(meta.origin.eq("funchal").to_numpy())
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    E = np.zeros((len(rows), 1536), np.float32)
    for k, i in enumerate(rows):
        r = meta.iloc[i]
        x = waves[r.offset:r.offset + r.length].astype(np.float32)
        c = np.stack(audio_chunks(whiten(x)))
        E[k] = session.run(["embedding"], {"inputs": c})[0].mean(0)
    np.savez_compressed(CACHE / "emb_perch_funchal.npz", rows=rows, embedding=E)
    print("saved perch funchal", E.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("encoder", choices=["perch", "ast", "clap", "biolingual"])
    enc = parser.parse_args().encoder
    perch_funchal() if enc == "perch" else run(enc)
