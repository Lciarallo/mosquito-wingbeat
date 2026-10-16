"""Non-recurrent classifiers for Aedes / other mosquito / noise recognition."""
import numpy as np
import pandas as pd
import torch
from torch import nn


class SpectrogramCNN(nn.Module):
    """Keep frequency location, pool over time; no recurrent prediction task."""
    def __init__(self, features=16, classes=3):
        super().__init__()
        if features < 4 or features % 2:
            raise ValueError('An even number of >=4 frequency bands is required.')
        self.config = dict(features=features, classes=classes)
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, (5, 3), padding=(2, 1)), nn.GroupNorm(4, 16), nn.GELU(),
            nn.AvgPool2d((3, 1)),
            nn.Conv2d(16, 32, (5, 3), padding=(2, 1)), nn.GroupNorm(8, 32), nn.GELU(),
            nn.AvgPool2d((3, 2)),
            nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * features, 128),
                                  nn.GELU(), nn.Dropout(.25), nn.Linear(128, classes))

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.config['features'] or x.shape[1] < 9:
            raise ValueError('Expected batch x time (>=9) x configured Mel bands.')
        z = self.features(x[:, None])
        return self.head(torch.cat([z.mean(2), z.amax(2)], dim=1))


class AcousticMLP(nn.Module):
    """Feed-forward neural classifier on per-window acoustic statistics."""
    def __init__(self, features=80, classes=3):
        super().__init__()
        self.config = dict(features=features, classes=classes)
        self.network = nn.Sequential(nn.Linear(features, 128), nn.LayerNorm(128),
            nn.GELU(), nn.Dropout(.25), nn.Linear(128, 64), nn.GELU(),
            nn.Dropout(.2), nn.Linear(64, classes))

    def forward(self, x):
        return self.network(x)


def source_class_weights(frame, gamma=.5):
    """Equal mass for observed (source, label) cells, from training rows only."""
    y = frame.window_target.to_numpy(int)
    count = frame.groupby('group').group.transform('size').to_numpy()
    base = np.minimum(count, 60).astype(float) ** (-gamma)
    cells = pd.MultiIndex.from_arrays([frame.domain.astype(str), y])
    keys, inv = np.unique(cells.to_numpy(), return_inverse=True)
    mass = np.bincount(inv, weights=base)
    result = base / mass[inv]
    return result * len(frame) / result.sum()


def source_class_loss(frame, probabilities):
    """Group log loss with equal mass per observed source/class cell."""
    from train_ternary_lstm import GroupPool
    pool = GroupPool(frame)
    p = pool.probabilities(probabilities)
    table = pd.DataFrame(dict(source=pool.sources, target=pool.truth))
    size = table.groupby(['source', 'target']).target.transform('size').to_numpy()
    loss = -np.log(p[np.arange(len(p)), pool.truth].clip(1e-12))
    return float(np.average(loss, weights=1 / size))


def source_class_score(groups, prediction):
    """Mean recall across source/class cells, not raw source-dominated accuracy."""
    table = groups[['source', 'true_label']].copy()
    table['correct'] = np.asarray(prediction) == groups.true_label.to_numpy()
    return float(table.groupby(['source', 'true_label']).correct.mean().mean())


def assert_partitions(frame, fit, validation, test):
    for a, b in [(fit, validation), (fit, test), (validation, test)]:
        for column in ['group', 'pcm_block']:
            if set(frame[column].iloc[a]) & set(frame[column].iloc[b]):
                raise ValueError(f'Leakage in {column}')
    if len(set(fit) | set(validation) | set(test)) != len(frame):
        raise ValueError('Partition coverage mismatch.')


def temporal_statistics(sequences):
    z = np.asarray(sequences)
    return np.concatenate([z.mean(1), z.std(1),
                            *np.quantile(z, [.1, .5, .9], axis=1)], axis=1).astype('float32')
