"""Deep Neural Network Architectures for Mosquito Wingbeat Classification.

Includes:
1. HarmonicSEResNet: Deep Residual CNN with Squeeze-and-Excitation (SE) Channel Attention
   and Dilated Vertical Convolutions for multi-octave harmonic binding.
2. DeepBioacousticHybridNet: Dual-Branch multi-modal architecture fusing deep spectro-temporal
   embeddings from 2D Log-Mel representations with explicit 138-dim bioacoustic invariant features.
3. End-to-end training pipeline with SpecAugment, AdamW, and Cosine Annealing.
"""
from __future__ import annotations
from pathlib import Path
import json, time
import numpy as np, pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from scipy.special import expit

from aedes_binary import targets, weights, ROOT
from large_audio_model import Spectrogram, MELS, SR, SAMPLES
from bioacoustic_invariance import BIOACOUSTIC_FEATURE_NAMES
from aedes_classical import FEATURE_NAMES as BASE_FEATURE_NAMES

DEEP_MODELS_DIR = ROOT / 'results/models_deep'
DEEP_MODELS_DIR.mkdir(parents=True, exist_ok=True)

class SqueezeExcitation(nn.Module):
    """Channel-wise attention block that emphasizes harmonic wingbeat bands."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w

class HarmonicResBlock(nn.Module):
    """Residual block with dilated frequency convolution and Squeeze-and-Excitation."""
    def __init__(self, in_channels: int, out_channels: int, pool: tuple[int, int] = (2, 2), freq_dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=(freq_dilation, 1),
                               dilation=(freq_dilation, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcitation(out_channels)
        self.skip = (nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
                     if in_channels != out_channels else nn.Identity())
        self.pool = nn.AvgPool2d(pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.relu(out + residual, inplace=True)
        return self.pool(out)

class HarmonicSEResNet(nn.Module):
    """Deep Squeeze-and-Excitation Residual Network for audio spectrograms."""
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.frontend = Spectrogram()
        self.encoder = nn.Sequential(
            HarmonicResBlock(2, 32, pool=(2, 2), freq_dilation=1),
            HarmonicResBlock(32, 64, pool=(2, 2), freq_dilation=2),   # Dilation captures 2xF0 harmonic distance
            HarmonicResBlock(64, 128, pool=(2, 2), freq_dilation=2),  # Multi-scale harmonic binding
            HarmonicResBlock(128, 256, pool=(2, 1), freq_dilation=1)
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 2))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(0.35)
        )
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, wave: torch.Tensor, augment: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.frontend(wave)
        if augment and self.training:
            # SpecAugment: random frequency & time masking
            fm = torch.arange(MELS, device=x.device).view(1, 1, -1, 1)
            start_f = torch.randint(0, MELS - 8, (len(x), 1, 1, 1), device=x.device)
            mask_f = (fm >= start_f) & (fm < start_f + 8)
            tm = torch.arange(x.shape[-1], device=x.device).view(1, 1, 1, -1)
            start_t = torch.randint(0, max(1, x.shape[-1] - 8), (len(x), 1, 1, 1), device=x.device)
            mask_t = (tm >= start_t) & (tm < start_t + 8)
            x = x.masked_fill(mask_f | mask_t, 0.0)

        emb = self.proj(self.pool(self.encoder(x)))
        logit = self.classifier(emb).squeeze(-1)
        return logit, emb

class BioacousticFeatureMLP(nn.Module):
    """Deep feature network for 138-dim engineered bioacoustic invariant features."""
    def __init__(self, in_features: int = 138, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(192, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)

class DeepBioacousticHybridNet(nn.Module):
    """Dual-branch hybrid network fusing deep spectrogram representation with bioacoustic features."""
    def __init__(self, bio_features_dim: int = 138):
        super().__init__()
        self.spectrogram_branch = HarmonicSEResNet(embedding_dim=256)
        self.bioacoustic_branch = BioacousticFeatureMLP(in_features=bio_features_dim, out_dim=128)
        
        # Gated Cross-Attention / Multimodal fusion
        fusion_dim = 256 + 128
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )
        self.head = nn.Linear(128, 1)

    def forward(self, wave: torch.Tensor, bio_feats: torch.Tensor, augment: bool = False) -> torch.Tensor:
        _, spec_emb = self.spectrogram_branch(wave, augment=augment)
        bio_emb = self.bioacoustic_branch(bio_feats)
        combined = torch.cat([spec_emb, bio_emb], dim=-1)
        fused = self.fusion(combined)
        return self.head(fused).squeeze(-1)

class MultisourceWaveDataset(Dataset):
    def __init__(self, wave_mmap, indices, bio_feats, targets, weights):
        self.wave_mmap = wave_mmap
        self.indices = indices
        self.bio_feats = bio_feats
        self.targets = targets
        self.weights = weights

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        wave_idx = self.indices[idx]
        w = np.array(self.wave_mmap[wave_idx], dtype=np.float32, copy=True)
        # Random crop to 15872 if longer
        if len(w) > SAMPLES:
            max_start = len(w) - SAMPLES
            st = np.random.randint(0, max_start + 1)
            w = w[st:st + SAMPLES]
        elif len(w) < SAMPLES:
            # Short buffer (4800 samples): repeat up to SAMPLES for fixed batch processing
            repeats = int(np.ceil(SAMPLES / len(w)))
            w = np.tile(w, repeats)[:SAMPLES]
        b = self.bio_feats[idx]
        y = self.targets[idx]
        weight = self.weights[idx]
        return (torch.from_numpy(w),
                torch.from_numpy(b),
                torch.tensor(y, dtype=torch.float32),
                torch.tensor(weight, dtype=torch.float32))

def train_deep_models(epochs: int = 12, batch_size: int = 64):
    print('=== TRAINING DEEP NEURAL NETWORKS (SE-ResNet & HybridNet) ===', flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('PyTorch using device:', device, flush=True)

    f = pd.read_csv(ROOT / 'results/multisource/segments.csv', low_memory=False)
    y = targets(f)

    base_feats = np.load(ROOT / 'data/cache/aedes_classical_features.npy')
    bio_feats = np.load(ROOT / 'data/cache/bioacoustic_invariance_features.npy')
    all_bio_feats = np.concatenate([base_feats, bio_feats], axis=1).astype(np.float32)

    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    tr_mask = f.split.eq('train') & f.quality_ok & (y >= 0)
    va_mask = f.split.eq('validation') & f.quality_ok & (y >= 0)
    ca_mask = f.split.eq('calibration') & f.quality_ok & (y >= 0)

    tr_idx = np.flatnonzero(tr_mask)
    va_idx = np.flatnonzero(va_mask)
    ca_idx = np.flatnonzero(ca_mask)

    all_bio_feats[tr_idx] = scaler.fit_transform(all_bio_feats[tr_idx])
    all_bio_feats[va_idx] = scaler.transform(all_bio_feats[va_idx])
    all_bio_feats[ca_idx] = scaler.transform(all_bio_feats[ca_idx])

    tw = weights(f.iloc[tr_idx])
    vw = weights(f.iloc[va_idx])
    cw = weights(f.iloc[ca_idx])

    # Memory map waves
    wave_long = np.load(ROOT / 'data/cache/multisource/long.npy', mmap_mode='r')
    wave_short = np.load(ROOT / 'data/cache/multisource/short.npy', mmap_mode='r')
    
class FastDataset(Dataset):
    def __init__(self, split_indices, sub_f, targets_arr, weights_arr, feats_arr, wave_long, wave_short):
        self.indices = split_indices
        self.sub_f = sub_f.reset_index(drop=True)
        self.targets = targets_arr
        self.weights_arr = weights_arr
        self.feats = feats_arr
        self.wave_long = wave_long
        self.wave_short = wave_short

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = self.sub_f.iloc[i]
        w_idx = int(row.wave_index)
        if row.bucket == 'long':
            w = np.array(self.wave_long[w_idx], dtype=np.float32, copy=True)
        else:
            w_s = np.array(self.wave_short[w_idx], dtype=np.float32, copy=True)
            repeats = int(np.ceil(SAMPLES / len(w_s)))
            w = np.tile(w_s, repeats)[:SAMPLES]
        return (torch.from_numpy(w),
                torch.from_numpy(self.feats[i]),
                torch.tensor(self.targets[i], dtype=torch.float32),
                torch.tensor(self.weights_arr[i], dtype=torch.float32))

def train_deep_models(epochs: int = 10, batch_size: int = 64):
    print('=== TRAINING DEEP NEURAL NETWORKS (SE-ResNet & HybridNet) ===', flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('PyTorch using device:', device, flush=True)

    f = pd.read_csv(ROOT / 'results/multisource/segments.csv', low_memory=False)
    y = targets(f)

    base_feats = np.load(ROOT / 'data/cache/aedes_classical_features.npy')
    bio_feats = np.load(ROOT / 'data/cache/bioacoustic_invariance_features.npy')
    all_bio_feats = np.concatenate([base_feats, bio_feats], axis=1).astype(np.float32)

    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    tr_mask = f.split.eq('train') & f.quality_ok & (y >= 0)
    va_mask = f.split.eq('validation') & f.quality_ok & (y >= 0)
    ca_mask = f.split.eq('calibration') & f.quality_ok & (y >= 0)

    tr_idx = np.flatnonzero(tr_mask)
    va_idx = np.flatnonzero(va_mask)
    ca_idx = np.flatnonzero(ca_mask)

    all_bio_feats[tr_idx] = scaler.fit_transform(all_bio_feats[tr_idx])
    all_bio_feats[va_idx] = scaler.transform(all_bio_feats[va_idx])
    all_bio_feats[ca_idx] = scaler.transform(all_bio_feats[ca_idx])

    tw = weights(f.iloc[tr_idx])
    vw = weights(f.iloc[va_idx])
    cw = weights(f.iloc[ca_idx])

    # Memory map waves
    wave_long = np.load(ROOT / 'data/cache/multisource/long.npy', mmap_mode='r')
    wave_short = np.load(ROOT / 'data/cache/multisource/short.npy', mmap_mode='r')

    train_ds = FastDataset(tr_idx, f.iloc[tr_idx], y[tr_idx], tw, all_bio_feats[tr_idx], wave_long, wave_short)
    val_ds = FastDataset(va_idx, f.iloc[va_idx], y[va_idx], vw, all_bio_feats[va_idx], wave_long, wave_short)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # 1. Train DeepBioacousticHybridNet
    print('\nTraining DeepBioacousticHybridNet (Spectrogram SE-ResNet + 138-dim Bioacoustic MLP)...', flush=True)
    model = DeepBioacousticHybridNet(bio_features_dim=all_bio_feats.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    best_val_auc = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, steps = 0.0, 0
        t0 = time.time()
        for waves, b_feats, targets_batch, weights_batch in train_loader:
            waves = waves.to(device, non_blocking=True)
            b_feats = b_feats.to(device, non_blocking=True)
            targets_batch = targets_batch.to(device, non_blocking=True)
            weights_batch = weights_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(waves, b_feats, augment=True)
            loss_unweighted = criterion(logits, targets_batch)
            loss = (loss_unweighted * weights_batch).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1
        scheduler.step()

        # Validation
        model.eval()
        val_preds, val_targets, val_weights_list = [], [], []
        with torch.no_grad():
            for waves, b_feats, targets_batch, weights_batch in val_loader:
                waves = waves.to(device, non_blocking=True)
                b_feats = b_feats.to(device, non_blocking=True)
                logits = model(waves, b_feats, augment=False)
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_targets.extend(targets_batch.numpy())
                val_weights_list.extend(weights_batch.numpy())

        val_auc = float(roc_auc_score(val_targets, val_preds, sample_weight=val_weights_list))
        elapsed = time.time() - t0
        print(f'Epoch {epoch:2d}/{epochs} | Train Loss: {total_loss/steps:.4f} | Val AUC: {val_auc:.4f} | {elapsed:.1f}s', flush=True)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f'Best Validation AUC for DeepBioacousticHybridNet: {best_val_auc:.4f}', flush=True)
    model.load_state_dict(best_weights)
    model.to(device)

    # Calibrate on calibration set
    cal_ds = FastDataset(ca_idx, f.iloc[ca_idx], y[ca_idx], cw, all_bio_feats[ca_idx], wave_long, wave_short)
    cal_loader = DataLoader(cal_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    cal_logits_list, cal_targets_list, cal_weights_list = [], [], []
    model.eval()
    with torch.no_grad():
        for waves, b_feats, targets_batch, weights_batch in cal_loader:
            waves = waves.to(device, non_blocking=True)
            b_feats = b_feats.to(device, non_blocking=True)
            logits = model(waves, b_feats, augment=False)
            cal_logits_list.extend(logits.cpu().numpy())
            cal_targets_list.extend(targets_batch.numpy())
            cal_weights_list.extend(weights_batch.numpy())

    cal_z = np.array(cal_logits_list)
    cal_y = np.array(cal_targets_list)
    cal_w = np.array(cal_weights_list)

    from sklearn.linear_model import LogisticRegression
    calibrator = LogisticRegression(C=10, max_iter=2000).fit(cal_z[:, None], cal_y, sample_weight=cal_w)
    slope = float(calibrator.coef_[0, 0])
    bias = float(calibrator.intercept_[0])
    cal_p = expit(slope * cal_z + bias)

    high = next(float(t) for t in np.r_[np.linspace(0.5, 0.999, 500), 1.0001]
                if np.average(cal_p[cal_y == 0] >= t, weights=cal_w[cal_y == 0]) <= 0.05)
    low = 0.0
    for t in np.linspace(0.001, min(0.499, high - 1e-6), 499):
        if np.average(cal_p[cal_y == 1] <= t, weights=cal_w[cal_y == 1]) <= 0.05:
            low = float(t)

    checkpoint = {
        'state_dict': model.state_dict(),
        'val_auc': best_val_auc,
        'slope': slope,
        'bias': bias,
        'low': low,
        'high': high,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'family': 'deep_hybrid_net'
    }
    torch.save(checkpoint, DEEP_MODELS_DIR / 'deep_hybrid_net.pt')
    print(f'Saved deep checkpoint to {DEEP_MODELS_DIR / "deep_hybrid_net.pt"} (Calibrated thresholds: low={low:.4f}, high={high:.4f})', flush=True)

if __name__ == '__main__':
    train_deep_models(epochs=12, batch_size=64)
