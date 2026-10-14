"""Computer-only spectral, contrastive and simulated-swarm experiments.

Peak candidates are descriptive features, not validated mosquito identities or
temperature-invariant fundamentals. No private/evaluation audio is mixed.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from reproduction_models import normalize, ResidualBlock

SR = 8000
FFT = 2048
HOP = 128


def spectrum(wave):
    return torch.stft(normalize(wave.float()), FFT, hop_length=HOP,
                      window=torch.hann_window(FFT, device=wave.device),
                      center=False, return_complex=True).abs().square()


class RidgeFrontend(nn.Module):
    """Multiple spectral ridges retained; fixed 200--3500 Hz analysis band.

    Frequency-local and time-local contrasts reduce smooth channel/background
    contributions. They do not guarantee channel invariance or source separation.
    STFT uses only complete real frames, without reflection or waveform extension.
    """
    def forward(self, wave):
        power = spectrum(wave)
        hz = torch.arange(power.shape[1], device=wave.device) * SR / FFT
        power = power[:, (hz >= 200) & (hz <= 3500)]
        db = 10 * power.clamp_min(1e-10).log10()
        db = (db - db.amax((1, 2), keepdim=True)).clamp(-70, 0)
        base = db[:, None]
        local_frequency = F.avg_pool2d(base, (25, 1), stride=1,
                                      padding=(12, 0), count_include_pad=False)
        local_time = F.avg_pool2d(base, (1, 9), stride=1,
                                 padding=(0, 4), count_include_pad=False)
        image = torch.cat([(base + 70) / 70,
                           (base - local_frequency).clamp(-30, 30) / 20,
                           (base - local_time).clamp(-30, 30) / 20], 1)
        # Resize the representation, not the waveform.
        return F.interpolate(image, size=(160, 48), mode='bilinear', align_corners=False)


class StrategyNet(nn.Module):
    def __init__(self, taxa=36):
        super().__init__()
        self.frontend = RidgeFrontend()
        self.encoder = nn.Sequential(ResidualBlock(3, 64, (2, 2)),
                                     ResidualBlock(64, 128, (2, 2)),
                                     ResidualBlock(128, 256, (2, 2)),
                                     ResidualBlock(256, 384, (2, 1)))
        self.project = nn.Sequential(nn.AdaptiveAvgPool2d((5, 2)), nn.Flatten(),
                                     nn.Linear(384 * 10, 512), nn.LayerNorm(512),
                                     nn.ReLU(), nn.Dropout(.25))
        self.binary = nn.Linear(512, 1)
        self.species = nn.Linear(512, taxa)
        self.contrast = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128))

    def forward(self, wave):
        emb = self.project(self.encoder(self.frontend(wave)))
        return self.binary(emb).squeeze(1), self.species(emb), F.normalize(self.contrast(emb), dim=1)


def cross_domain_contrast(z, species, domain, group, temperature=.1):
    """Same taxon, different recording AND domain; never merge all negatives."""
    same = species[:, None].eq(species[None, :])
    diff_domain = ~domain[:, None].eq(domain[None, :])
    diff_group = ~group[:, None].eq(group[None, :])
    eye = torch.eye(len(z), device=z.device, dtype=torch.bool)
    positive = same & diff_domain & diff_group & ~eye
    valid = positive.any(1)
    if not valid.any():
        return z.sum() * 0, 0
    scores = F.normalize(z, dim=1) @ F.normalize(z, dim=1).T / temperature
    logp = scores - scores.masked_fill(eye, -1e4).logsumexp(1, keepdim=True)
    loss = -(logp.masked_fill(~positive, 0).sum(1) / positive.sum(1).clamp_min(1))[valid].mean()
    return loss, int(valid.sum())


def conditional_mmd(z, species, domain):
    """Biased RBF MMD estimate, conditioned on exact known taxon."""
    result = []
    for label in species.unique():
        pools = [z[species.eq(label) & domain.eq(d)] for d in domain[species.eq(label)].unique()]
        if len(pools) != 2 or min(len(a) for a in pools) < 2:
            continue
        a, b = pools
        aa = torch.cdist(a, a).square(); bb = torch.cdist(b, b).square(); ab = torch.cdist(a, b).square()
        value = sum(((-aa / t).exp().mean() + (-bb / t).exp().mean() -
                     2 * (-ab / t).exp().mean()) for t in [.25, 1., 4.]) / 3
        result.append(value)
    return torch.stack(result).mean() if result else z.sum() * 0


def mix_known_species(wave, species, group, generator=None, probability=.5):
    """Superposition only from same exact taxon and different original group.

    Preserves the supervised taxon, unlike mixing arbitrary non-Aedes taxa and
    assigning the anchor species. No temporal tiling; equal real-duration crops.
    """
    out = wave.clone(); pairs = []
    for i in range(len(wave)):
        candidates = torch.nonzero(species.eq(species[i]) & ~group.eq(group[i])).flatten()
        if len(candidates) and torch.rand((), device=wave.device, generator=generator) < probability:
            j = candidates[torch.randint(len(candidates), (), device=wave.device, generator=generator)]
            gain = .25 + .75 * torch.rand((), device=wave.device, generator=generator)
            out[i] += gain * wave[j]
            pairs.append((i, int(j)))
    return normalize(out), pairs


FEATURE_NAMES = ([f'peak_{rank}_{stat}' for rank in range(1, 5)
                  for stat in ['hz_median', 'hz_iqr', 'hz_step', 'prominence']]
                 + [f'harmonic_candidate_{rank}_{stat}' for rank in range(1, 4)
                    for stat in ['hz_median', 'hz_iqr', 'hz_step', 'support']]
                 + ['entropy_median', 'entropy_iqr', 'peak_count_median',
                    'peak_count_iqr', 'harmonic_support_median', 'harmonic_support_iqr'])


@torch.no_grad()
def multipitch_features(wave):
    """34 descriptors retaining four peaks and three harmonic candidates/frame.

    Ranking can switch over time; step size is not a validated individual track.
    Absolute power removed by per-frame normalization, not temperature effects.
    """
    power = spectrum(wave)
    hz = torch.arange(power.shape[1], device=wave.device) * SR / FFT
    band = (hz >= 200) & (hz <= 900)
    db = 10 * power.clamp_min(1e-10).log10()
    smooth = F.avg_pool1d(db.transpose(1, 2).reshape(-1, 1, len(hz)), 25,
                        stride=1, padding=12, count_include_pad=False)
    smooth = smooth.reshape(len(wave), -1, len(hz)).transpose(1, 2)
    prominence = (db - smooth).clamp_min(0)
    local = (power >= F.max_pool1d(power.transpose(1, 2).reshape(-1, 1, len(hz)),
                                  9, stride=1, padding=4).reshape(len(wave), -1, len(hz)).transpose(1, 2))
    values, ix = prominence[:, band].masked_fill(~local[:, band], -1).topk(4, dim=1)
    freq = hz[band][ix]
    valid = values >= 3
    freq = torch.where(valid, freq, torch.zeros_like(freq))
    values = values.clamp_min(0)
    candidates = hz[band]; score = torch.zeros_like(prominence[:, band])
    for h in [1, 2, 3, 4]:
        bins = (candidates * h / (SR / FFT)).round().long()
        score += prominence[:, bins] / h
    support, ci = score.topk(3, dim=1)
    cfreq = candidates[ci]
    def describe(freqs, strengths):
        out = []
        for k in range(freqs.shape[1]):
            a = freqs[:, k]; b = strengths[:, k]
            out += [a.median(1).values, a.quantile(.75, dim=1) - a.quantile(.25, dim=1),
                    a.diff(dim=1).abs().median(1).values if a.shape[1] > 1 else a[:, 0] * 0,
                    b.median(1).values]
        return out
    relative = power[:, band] / power[:, band].sum(1, keepdim=True).clamp_min(1e-10)
    entropy = -(relative * relative.clamp_min(1e-10).log()).sum(1) / np.log(int(band.sum()))
    counts = (local[:, band] & (prominence[:, band] >= 6)).sum(1).float()
    best = support[:, 0]
    stats = describe(freq, values) + describe(cfreq, support)
    for a in [entropy, counts, best]:
        stats += [a.median(1).values, a.quantile(.75, dim=1) - a.quantile(.25, dim=1)]
    return torch.stack(stats, 1)
