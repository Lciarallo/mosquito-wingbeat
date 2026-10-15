"""Offline transfer of the supplied notebook's 61 features and floor-relative LFCC.

Each call consumes exactly one 1 s window at 16 kHz; no recording or batch gain
normalisation. The received notebook itself is never executed as a whole.
"""
from __future__ import annotations
import ast
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
from scipy.signal import butter, sosfilt
from scipy.fft import rfft, rfftfreq, dct
import acoustic_features_v4 as frozen_v4

SR = 16000
ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / 'external/notebook_reference_2026_10_14/Mosquito_v2.ipynb'
NAMES = ['time_rms', 'time_zcr', 'time_peak', 'time_crest_factor', 'spec_f0_wingbeat',
         'spec_centroid', 'spec_bandwidth', 'spec_rolloff', 'spec_flatness']
NAMES += [f'{scale}_{stat}_{i}' for i in range(1, 14) for scale, stat in
          [('mfcc', 'mean'), ('mfcc', 'std'), ('lfcc', 'mean'), ('lfcc', 'std')]]
LFCC_NAMES = [f'lfcc_relative_mean_{i}' for i in range(13)] + [f'lfcc_relative_std_{i}' for i in range(13)]
_ALLOWED = {'remover_ruido_passa_faixa', 'normalizar_amplitude', 'pre_processar_audio',
            '_compute_stft_frames', '_get_filterbank', '_extract_cepstral_stats', 'extract_features_1s'}


def load_reference_functions(path=REFERENCE, cache_filterbank=True):
    notebook = json.loads(Path(path).read_text())
    definitions = []
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        for node in ast.parse(''.join(cell.get('source', []))).body:
            if isinstance(node, ast.FunctionDef) and node.name in _ALLOWED:
                definitions.append(node)
    if {n.name for n in definitions} != _ALLOWED:
        raise ValueError('Received notebook feature API differs from audited source.')
    scope = dict(np=np, butter=butter, sosfilt=sosfilt, rfft=rfft, rfftfreq=rfftfreq, dct=dct)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), scope)
    if cache_filterbank:
        scope['_get_filterbank'] = lru_cache(maxsize=8)(scope['_get_filterbank'])
    return scope


@lru_cache(maxsize=1)
def _reference():
    return load_reference_functions()


def checked_window(raw_wave):
    wave = np.asarray(raw_wave)
    if wave.ndim != 1 or len(wave) != SR or not np.isfinite(wave).all():
        raise ValueError('Exactly one finite mono 1 s window at 16 kHz is required.')
    return wave


def reference_window_features(raw_wave):
    wave = checked_window(raw_wave)
    scope = _reference()
    conditioned = scope['pre_processar_audio'](wave, SR)
    features = scope['extract_features_1s'](conditioned, SR)
    if list(features) != NAMES:
        raise ValueError('Reference feature order changed.')
    result = np.array([features[k] for k in NAMES], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite reference features.')
    return result


def _linear_bank():
    f = frozen_v4.SHORT_F
    points = np.linspace(200., 3900., 28)
    bank = np.maximum(0., np.minimum((f[None, :] - points[:-2, None]) / np.diff(points)[:-1, None],
                                    (points[2:, None] - f[None, :]) / np.diff(points)[1:, None]))
    return bank / np.maximum(bank.sum(axis=1, keepdims=True), 1e-12)


LFCC_BANK = _linear_bank()


def relative_lfcc_features(raw_wave):
    wave = checked_window(raw_wave).astype(np.float64)
    wave = wave - wave.mean()
    wave = wave / max(float(np.sqrt(np.mean(wave * wave))), 1e-8)
    _, _, power = frozen_v4.stft_power(wave, frozen_v4.SHORT_NPERSEG, frozen_v4.SHORT_HOP)
    floor, _ = frozen_v4.local_noise_floor(power, 10.)
    db = 10. * np.log10(np.maximum(LFCC_BANK @ (power / np.maximum(floor, 1e-12)), 1e-6))
    db -= np.median(db, axis=0, keepdims=True)
    coefficients = dct(db, type=2, axis=0, norm='ortho')[:13]
    means = coefficients.mean(axis=1)
    means[0] = 0.
    result = np.r_[means, coefficients.std(axis=1)].astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite relative LFCC.')
    return result


def extract_features(windows):
    rows = [(reference_window_features(w), relative_lfcc_features(w)) for w in windows]
    if not rows:
        raise ValueError('Empty window batch.')
    return np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows])


def hybrid_window_features(raw_wave):
    base, salience = frozen_v4.window_features(checked_window(raw_wave))
    return np.r_[base, relative_lfcc_features(raw_wave)].astype(np.float32), salience


def pool_features(features, salience, fraction=1.):
    """Deterministic bag aggregation; pooling never fits a batch normaliser.

    All compared variants use the same fixed arithmetic mean in this experiment,
    so LFCC's effect is not confounded with changing the aggregation policy.
    Selective pooling is available for future calibration-only experiments.
    """
    x, s = np.asarray(features), np.asarray(salience)
    if x.ndim != 2 or len(x) == 0 or len(s) != len(x) or not 0 < fraction <= 1:
        raise ValueError('Invalid recording bag.')
    if fraction == 1.:
        return np.mean(x.astype(np.float64), axis=0)
    n = len(x) if len(x) <= 4 else max(1, int(np.ceil(len(x) * fraction)))
    indices = np.argsort(-s, kind='stable')[:n]
    weights = np.maximum(s[indices], 1e-12)
    return np.average(x[indices].astype(np.float64), axis=0, weights=weights)
