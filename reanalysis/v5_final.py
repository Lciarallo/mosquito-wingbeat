"""v5 (exploratória): congela o candidato Perch + embedding óptico e aplica em arquivos novos.

  freeze   Ajusta as duas cabeças logísticas (Perch normalizado; embedding da CNN óptica com aumento) em todo o
           desenvolvimento acústico (Abuzz, HumBug, MosquitoSong, Funchal e BDC restante). Também escolhe um
           limiar de pesquisa (FPR <= 2% em outros mosquitos, por validação por cluster) e registra os SHA-256.
  predict  Pontua WAVs novos sem retreinar nada: média dos logits das duas cabeças.

Uso:
  .venv-v3/bin/python -m reanalysis.v5_final freeze
  .venv-v3/bin/python -m reanalysis.v5_final predict audio1.wav [audio2.wav ...]

O limiar NÃO é operacional: no LODO, nenhum limiar escolhido no treino se transferiu (recall <= 12%).
Para um aparelho novo, recalibre (e de preferência retreine as cabeças) com clusters gravados nele.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from scipy.signal import sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results/v5/final"
PUBLIC = ROOT / "reanalysis/outputs/v5"
CNN = ROOT / "data/cache/v5/optical_cnn_aug.pt"


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def freeze():
    from sklearn.model_selection import StratifiedGroupKFold
    from reanalysis import v5_eval as ev
    meta, reps, _, _ = ev.load()
    y = (meta.target.to_numpy() == 2).astype(int)
    origin = meta.origin.to_numpy().astype(object)
    groups = meta.cluster_merged.astype(str).to_numpy()
    names = ["A0_perch", "H1b_optico_embedding_aug"]
    idx = np.arange(len(y))
    oof = np.zeros(len(y))
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=ev.SEED).split(idx, y, groups):
        oof[b] = np.mean([ev.fit_score(reps[n], y, origin, a, b) for n in names], 0)
    threshold = float(np.quantile(oof[y == 0], .98))
    heads = {}
    for n in names:
        m = ev.head(reps[n].shape[1])
        m.fit(reps[n], y, logisticregression__sample_weight=ev.source_weights(origin, y))
        heads[n] = m
    FINAL.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(heads=heads, threshold=threshold, names=names), FINAL / "heads.joblib")
    from reanalysis.v3_features import MODEL
    record = dict(candidato="Perch normalizado + embedding da CNN óptica (média dos logits)", criado_em="2026-11-01",
                  limiar_pesquisa_fpr2=threshold, n_treino=dict(aedes=int(y.sum()), outros=int((1 - y).sum())),
                  corpora=sorted(set(origin)),
                  sha256={"results/v5/final/heads.joblib": digest(FINAL / "heads.joblib"),
                          "data/cache/v5/optical_cnn_aug.pt": digest(CNN),
                          str(MODEL.relative_to(ROOT)): digest(MODEL)},
                  aviso="Exploratório. Sem teste novo; o limiar não se transferiu entre corpora no LODO.")
    PUBLIC.mkdir(parents=True, exist_ok=True)
    (PUBLIC / "final_freeze.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(record, indent=1, ensure_ascii=False))


def embed_file(path, session, cnn):
    import torch
    from reanalysis import frontend as fe
    from reanalysis.v3_embed import whiten
    from reanalysis.v3_features import audio_chunks
    from reanalysis.v5_optical import HOP_WIN, MAX_WIN, WIN, spectrogram
    x = fe.decode(Path(path))
    if len(x) > 30:
        x = sosfiltfilt(fe.SOS, x).astype(np.float32)
    perch = session.run(["embedding"], {"inputs": np.stack(audio_chunks(whiten(x)))})[0].mean(0)
    if len(x) < WIN:
        pad = WIN - len(x)
        x = np.pad(x, (pad // 2, pad - pad // 2))
    starts = np.arange(0, len(x) - WIN + 1, HOP_WIN)
    if len(starts) > MAX_WIN:
        starts = starts[np.unique(np.linspace(0, len(starts) - 1, MAX_WIN).round().astype(int))]
    w = np.stack([x[s:s + WIN] for s in starts]).astype(np.float32)
    keep = np.argsort(-np.sqrt((w ** 2).mean(1)))[:max(1, int(np.ceil(len(w) / 2)))]
    with torch.no_grad():
        optical = cnn.features(spectrogram(torch.from_numpy(w[keep]))).mean(0).numpy()
    return perch, optical


def predict(paths):
    import onnxruntime as ort
    import torch
    from reanalysis import v5_optical as vo
    from reanalysis.v3_features import MODEL
    record = json.loads((PUBLIC / "final_freeze.json").read_text())
    for rel, sha in record["sha256"].items():
        if digest(ROOT / rel) != sha:
            raise ValueError(f"Arquivo congelado mudou: {rel}")
    bundle = joblib.load(FINAL / "heads.joblib")
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    ck = torch.load(CNN, map_location="cpu")
    cnn = vo.Net(len(ck["classes"]))
    cnn.load_state_dict(ck["state"])
    cnn.eval()
    for path in paths:
        perch, optical = embed_file(path, session, cnn)
        h = bundle["heads"]
        logit = float(np.mean([h["A0_perch"].decision_function(perch[None])[0],
                               h["H1b_optico_embedding_aug"].decision_function(optical[None])[0]]))
        print(json.dumps(dict(arquivo=str(path), logit_aedes=round(logit, 4),
                              acima_do_limiar_de_pesquisa=bool(logit > bundle["threshold"])), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze", "predict"])
    parser.add_argument("paths", nargs="*")
    a = parser.parse_args()
    if a.stage == "freeze":
        freeze()
    else:
        if not a.paths:
            sys.exit("informe ao menos um WAV")
        predict(a.paths)
