"""Audit metrics; frozen historical evaluation modules remain unchanged.

Invalid audio stays in the sensitivity/specificity denominators. Undefined
precision and bootstrap draws are reported explicitly, never replaced by zero.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def validate(y, scores, quality, threshold):
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    quality = np.asarray(quality)
    if any(a.ndim != 1 for a in (y, scores, quality)) or not (len(y) == len(scores) == len(quality)):
        raise ValueError('Labels, scores and quality must be aligned vectors.')
    if not np.isin(y, [-1, 0, 1]).all():
        raise ValueError('Labels must be -1 (unknown), 0 or 1.')
    if not np.isin(quality, [False, True]).all():
        raise ValueError('Quality must contain booleans or 0/1.')
    if not np.isfinite(threshold):
        raise ValueError('Threshold must be finite.')
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Scores must be finite probabilities in [0, 1].')
    return y, scores, quality.astype(bool)


def _divide(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    return np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator > 0)


def _metrics(counts):
    tn, fp, neg_invalid, fn, tp, pos_invalid = np.asarray(counts, dtype=float).T
    pos, neg = fn + tp + pos_invalid, tn + fp + neg_invalid
    precision = _divide(tp, tp + fp)
    recall, specificity = _divide(tp, pos), _divide(tn, neg)
    f1 = _divide(2 * tp, tp + fp + pos)
    f1 = np.where(pos > 0, f1, np.nan)
    negative_f1 = _divide(2 * tn, tn + fn + neg)
    both = (pos > 0) & (neg > 0)
    return dict(aedes_precision=precision, aedes_recall=recall, specificity=specificity,
                aedes_f1=f1, macro_f1=np.where(both, (f1 + negative_f1) / 2, np.nan),
                balanced_accuracy=np.where(both, (recall + specificity) / 2, np.nan),
                emitted_false_positive_rate=_divide(fp, neg),
                coverage=_divide(tn + fp + fn + tp, pos + neg),
                accuracy=_divide(tn + tp, pos + neg))


def _counts(y, scores, quality, threshold):
    pred = np.where(quality, (scores >= threshold).astype(int), 2)
    return np.column_stack([(y == label) & (pred == decision)
                            for label in (0, 1) for decision in (0, 1, 2)])


def binary_metrics(y, scores, quality, threshold):
    y, scores, quality = validate(y, scores, quality, threshold)
    known = y >= 0
    counts = _counts(y[known], scores[known], quality[known], threshold).sum(0)
    result = {name: float(value) if np.isfinite(value) else None for name, value in _metrics(counts).items()}
    result.update(known_windows=int(known.sum()), unknown_windows=int((~known).sum()),
                  confusion=counts.reshape(2, 3).tolist(),
                  confusion_rows=['not_aedes', 'aedes'], confusion_columns=['not_aedes', 'aedes', 'invalid'],
                  precision_informative_for_binary_detection=bool(np.unique(y[known]).size == 2 and counts[4] + counts[1] > 0))
    both = np.unique(y[known]).size == 2
    result['roc_auc'] = float(roc_auc_score(y[known], scores[known])) if both else None
    result['average_precision'] = float(average_precision_score(y[known], scores[known])) if both else None
    result['ranking_scope'] = 'Raw scores on known labels; quality-invalid audio remains included.'
    return result


def cluster_intervals(y, scores, quality, groups, threshold=.5, replicates=2000, seed=20261007):
    """Percentile bootstrap of complete recording groups, conditional on this source.

    No inference to unseen devices or biological individuals is implied. Undefined
    resamples are omitted from percentiles and counted. Fewer than two recording
    groups provides no estimable recording-level uncertainty.
    """
    y, scores, quality = validate(y, scores, quality, threshold)
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) != len(y) or replicates < 1:
        raise ValueError('Aligned groups and positive replicate count required.')
    if any(g is None or str(g) in ('', 'nan', '<NA>') for g in groups):
        raise ValueError('Recording group identifiers are required.')
    known = y >= 0
    unique, inverse = np.unique(groups[known], return_inverse=True)
    counts = np.zeros((len(unique), 6))
    np.add.at(counts, inverse, _counts(y[known], scores[known], quality[known], threshold))
    rng = np.random.default_rng(seed)
    totals = counts[rng.integers(len(unique), size=(replicates, len(unique)))].sum(1) if len(unique) else np.zeros((replicates, 6))
    point = _metrics(counts.sum(0))
    output = {}
    for name, values in _metrics(totals).items():
        finite = values[np.isfinite(values)]
        estimable = len(unique) >= 2 and len(finite) >= 2 and np.isfinite(point[name])
        output[name] = dict(lower95=float(np.quantile(finite, .025)) if estimable else None,
                            upper95=float(np.quantile(finite, .975)) if estimable else None,
                            valid_replicates=len(finite), undefined_replicates=replicates-len(finite))
    return dict(method='Percentile bootstrap of whole recording groups within one source.',
                groups=len(unique), replicates=replicates, seed=seed, metrics=output)
