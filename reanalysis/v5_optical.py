"""v5 (exploratória), H1: o Wingbeats óptico (265 mil clipes) ensina *Ae. aegypti* para o microfone?

  pretrain   CNN de 6 espécies só no óptico, com aumento (equalização aleatória e ruído do OpenSLR 28).
             Também uma versão sem aumento, como diagnóstico.
  apply      Aplica as CNNs ópticas aos arquivos acústicos: pontuação direta (H1a) e embedding de 128 (H1b).
  lodo       Para cada corpus acústico fora: CNN do zero no acústico (C0) e CNN óptica afinada no acústico (H1c).

Entrada comum: 8 kHz, janelas de 5.000 amostras (0,625 s, o tamanho do clipe óptico), log-espectro de 94 a
3.406 Hz, padronizado por janela. No acústico, janelas com salto de 2.500, no máximo 64 por arquivo, e só a
metade de maior energia de cada arquivo entra (sem rótulo). Pontuação do arquivo = média nas janelas.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
import torch  # noqa: E402
from torch import nn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache/v5"
OUT = ROOT / "reanalysis/outputs/v5"
SLR = ROOT / "data/external/openslr28/extracted/RIRS_NOISES"
SEED = 20261102
WIN, HOP_WIN, MAX_WIN = 5000, 2500, 64
N_FFT, HOP = 256, 64
BINS = slice(3, 110)  # 94-3.406 Hz
LODO = ["abuzz", "humbug", "mosquitosong", "funchal"]
DEV = "cuda"


def spectrogram(x):
    """x: (B, 5000) float -> (B, 1, 107, 79), log-potência padronizada por janela."""
    x = x - x.mean(1, keepdim=True)
    S = torch.stft(x, N_FFT, HOP, window=torch.hann_window(N_FFT, device=x.device), return_complex=True)
    P = torch.log(S.abs()[:, BINS] ** 2 + 1e-8)
    P = (P - P.mean((1, 2), keepdim=True)) / (P.std((1, 2), keepdim=True) + 1e-5)
    return P[:, None]


class Net(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        def block(i, o, pool):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(pool))
        self.trunk = nn.Sequential(block(1, 32, (2, 2)), block(32, 64, (2, 2)), block(64, 128, (1, 2)),
                                   nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.embed = nn.Sequential(nn.Flatten(), nn.Dropout(.3), nn.Linear(128 * 26, 128), nn.ReLU())
        self.head = nn.Linear(128, n_out)

    def features(self, spec):
        return self.embed(self.trunk(spec).mean(3))  # média no tempo; a posição na frequência é mantida

    def forward(self, spec):
        return self.head(self.features(spec))


def noise_bank(n_files=400):
    real = sorted((SLR / "real_rirs_isotropic_noises").glob("*.wav"))
    files = [p for p in real if "noise" in p.name] + sorted((SLR / "pointsource_noises").glob("*.wav"))
    rng = np.random.default_rng(SEED)
    out = []
    for p in rng.choice(files, min(n_files, len(files)), replace=False):
        x, rate = sf.read(str(p), dtype="float32", always_2d=True)
        x = x[:, 0]
        if rate != 8000:
            g = np.gcd(rate, 8000)
            x = resample_poly(x, 8000 // g, rate // g).astype(np.float32)
        if len(x) > WIN:
            out.append(x / (np.sqrt(np.mean(x ** 2)) + 1e-8))
    return torch.from_numpy(np.concatenate(out)).to(DEV)


def augment(x, noise, gen):
    """Ruído externo (SNR 0-30 dB, p = 0,8) na forma de onda; a equalização é aplicada no espectro."""
    B = len(x)
    rms = x.pow(2).mean(1, keepdim=True).sqrt() + 1e-8
    starts = torch.randint(0, len(noise) - WIN, (B,), device=DEV, generator=gen)
    n = noise[starts[:, None] + torch.arange(WIN, device=DEV)]
    snr = torch.rand(B, 1, device=DEV, generator=gen) * 30
    use = (torch.rand(B, 1, device=DEV, generator=gen) < .8).float()
    return x + use * n * rms / 10 ** (snr / 20)


def random_eq(spec, gen):
    """Curva suave de ganho na frequência (6 pontos, desvio 0,7 em log-potência, cerca de 3 dB), p = 0,8."""
    B, F = spec.shape[0], spec.shape[2]
    ctrl = torch.randn(B, 1, 6, device=DEV, generator=gen) * .7
    curve = nn.functional.interpolate(ctrl, size=F, mode="linear", align_corners=True)[:, :, :, None]
    use = (torch.rand(B, 1, 1, 1, device=DEV, generator=gen) < .8).float()
    out = spec + use * curve.transpose(1, 2).reshape(B, 1, F, 1)
    return (out - out.mean((1, 2, 3), keepdim=True)) / (out.std((1, 2, 3), keepdim=True) + 1e-5)


def pretrain(aug=True, epochs=4, batch=512):
    torch.manual_seed(SEED)
    gen = torch.Generator(device=DEV).manual_seed(SEED)
    meta = pd.read_csv(CACHE / "optical_meta.csv")
    classes = sorted(meta.species.unique())
    y_all = torch.tensor(meta.species.map({c: i for i, c in enumerate(classes)}).to_numpy(), device=DEV)
    waves = torch.from_numpy(np.load(CACHE / "optical_waves.npy")).to(DEV)
    rng = np.random.default_rng(SEED)
    val = torch.from_numpy(rng.random(len(meta)) < .05).to(DEV)
    tr_idx = torch.nonzero(~val).squeeze(1)
    counts = torch.bincount(y_all[tr_idx], minlength=len(classes)).float()
    weights = (1 / counts)[y_all[tr_idx]]
    noise = noise_bank() if aug else None
    model = Net(len(classes)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    steps = epochs * len(tr_idx) // batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    model.train()
    for step in range(steps):
        pick = tr_idx[torch.multinomial(weights, batch, replacement=True, generator=gen)]
        x = waves[pick].float() / 32768
        if aug:
            x = augment(x, noise, gen)
        s = spectrogram(x)
        if aug:
            s = random_eq(s, gen)
        loss = nn.functional.cross_entropy(model(s), y_all[pick])
        opt.zero_grad(), loss.backward(), opt.step(), sched.step()
        if step % 200 == 0:
            print(f"step {step}/{steps} loss {loss.item():.3f}", flush=True)
    model.eval()
    vi = torch.nonzero(val).squeeze(1)
    preds = []
    with torch.no_grad():
        for k in range(0, len(vi), 2048):
            preds.append(model(spectrogram(waves[vi[k:k + 2048]].float() / 32768)).argmax(1))
    pred = torch.cat(preds).cpu().numpy()
    yv = y_all[vi].cpu().numpy()
    bal = float(np.mean([(pred[yv == c] == c).mean() for c in range(len(classes))]))
    aed = classes.index("Aedes aegypti")
    tag = "aug" if aug else "noaug"
    torch.save(dict(state=model.state_dict(), classes=classes), CACHE / f"optical_cnn_{tag}.pt")
    info = dict(tag=tag, epochs=epochs, steps=steps, optico_validacao_acc_balanceada_6_especies=round(bal, 4),
                optico_recall_aegypti=round(float((pred[yv == aed] == aed).mean()), 4), n_validacao=int(len(vi)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"optical_pretrain_{tag}.json").write_text(json.dumps(info, indent=1, ensure_ascii=False) + "\n")
    print(info)


def acoustic_windows():
    """Janelas de cada arquivo acústico (metade de maior energia, no máximo 64 antes do corte)."""
    meta = pd.read_csv(CACHE / "acoustic_meta.csv")
    waves = np.load(CACHE / "acoustic_waves.npy", mmap_mode="r")
    wins, owner = [], []
    for i, r in enumerate(meta.itertuples()):
        x = waves[r.offset:r.offset + r.length].astype(np.float32)
        if len(x) < WIN:
            pad = WIN - len(x)
            x = np.pad(x, (pad // 2, pad - pad // 2))
        starts = np.arange(0, len(x) - WIN + 1, HOP_WIN)
        if len(starts) > MAX_WIN:
            starts = starts[np.unique(np.linspace(0, len(starts) - 1, MAX_WIN).round().astype(int))]
        w = np.stack([x[s:s + WIN] for s in starts])
        e = np.sqrt((w ** 2).mean(1))
        keep = np.argsort(-e)[:max(1, int(np.ceil(len(w) / 2)))]
        wins.append(w[keep]), owner.extend([i] * len(keep))
    return meta, np.concatenate(wins).astype(np.float32), np.array(owner)


def per_file(values, owner, n):
    out = np.zeros((n,) + values.shape[1:], np.float64)
    np.add.at(out, owner, values)
    return out / np.bincount(owner, minlength=n).reshape((-1,) + (1,) * (values.ndim - 1))


def run_model(model, W, mode):
    res = []
    with torch.no_grad():
        for k in range(0, len(W), 4096):
            s = spectrogram(torch.from_numpy(W[k:k + 4096]).to(DEV))
            res.append((model.features(s) if mode == "features" else model(s)).float().cpu().numpy())
    return np.concatenate(res)


def apply():
    meta, W, owner = acoustic_windows()
    out = {}
    for tag in ["aug", "noaug"]:
        ck = torch.load(CACHE / f"optical_cnn_{tag}.pt", map_location=DEV)
        model = Net(len(ck["classes"])).to(DEV)
        model.load_state_dict(ck["state"])
        model.eval()
        logits = run_model(model, W, "logits")
        aed = ck["classes"].index("Aedes aegypti")
        others = [c for c in range(len(ck["classes"])) if c != aed]
        z = logits[:, aed] - np.logaddexp.reduce(logits[:, others], axis=1)
        out[f"h1a_{tag}"] = per_file(z, owner, len(meta))
        out[f"h1b_{tag}"] = per_file(run_model(model, W, "features"), owner, len(meta))
    np.savez_compressed(CACHE / "h1_optical_applied.npz", **out)
    print({k: v.shape for k, v in out.items()})


def source_weights(origin, y):
    w = np.ones(len(y))
    for c in np.unique(y):
        mc = y == c
        S = np.unique(origin[mc])
        for s in S:
            k = mc & (origin == s)
            w[k] = 1 / k.sum() / len(S)
    return w / w.mean()


def lodo(epochs=12, per_epoch=20000, batch=256):
    meta, W, owner = acoustic_windows()
    y_file = (meta.target.to_numpy() == 2).astype(np.float32)
    origin = meta.origin.to_numpy().astype(object)
    Wt = torch.from_numpy(W).to(DEV)
    scores = {}
    for init in ["c0_zero", "h1c_optico"]:
        for held in LODO:
            torch.manual_seed(SEED)
            gen = torch.Generator(device=DEV).manual_seed(SEED)
            train_files = np.flatnonzero(origin != held)
            wf = np.zeros(len(meta))
            wf[train_files] = source_weights(origin[train_files], y_file[train_files])
            nwin = np.bincount(owner, minlength=len(meta))
            ww = torch.tensor(wf[owner] / nwin[owner], device=DEV, dtype=torch.float32)
            yw = torch.tensor(y_file[owner], device=DEV)
            model = Net(6).to(DEV)
            if init == "h1c_optico":
                model.load_state_dict(torch.load(CACHE / "optical_cnn_aug.pt", map_location=DEV)["state"])
            model.head = nn.Linear(128, 1).to(DEV)
            lr = 3e-4 if init == "h1c_optico" else 1e-3
            opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-4)
            model.train()
            for ep in range(epochs):
                for _ in range(per_epoch // batch):
                    pick = torch.multinomial(ww, batch, replacement=True, generator=gen)
                    s = random_eq(spectrogram(Wt[pick]), gen)
                    loss = nn.functional.binary_cross_entropy_with_logits(model(s)[:, 0], yw[pick])
                    opt.zero_grad(), loss.backward(), opt.step()
                print(init, held, ep, round(loss.item(), 3), flush=True)
            model.eval()
            test = np.flatnonzero(origin[owner] == held)
            z = run_model(model, W[test], "logits")[:, 0]
            s_file = per_file(z, owner[test], len(meta))
            scores[f"{init}|{held}"] = s_file
    np.savez_compressed(CACHE / "h1_lodo_scores.npz", **scores)
    print("saved", list(scores))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["pretrain", "apply", "lodo"])
    stage = parser.parse_args().stage
    if stage == "pretrain":
        pretrain(aug=True)
        pretrain(aug=False)
    elif stage == "apply":
        apply()
    else:
        lodo()
