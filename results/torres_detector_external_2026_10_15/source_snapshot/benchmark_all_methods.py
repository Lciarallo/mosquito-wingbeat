"""Comprehensive Unified Benchmark of All ML Methods Across Diverse Sources.

Evaluates:
- Baseline CNNs and Baseline Classifiers
- Modern Tree Ensembles (LightGBM, HistGB, ExtraTrees, RandomForest)
- Deep Neural Networks (DeepBioacousticHybridNet)
- Committee Ensembles (Soft-Voting, Stacking Meta-Learner)

Sources tested:
1. Dryad (7901 windows)
2. HumBug (1510 windows)
3. MosquitoSong (352 windows)
4. UFRGS Out-of-Domain (417 windows)
5. Private Swarm Audio (107 windows)
6. Recording-disjoint held-out noise; 436 contaminated controls excluded.
"""
from pathlib import Path
import json, time
import numpy as np, pandas as pd, joblib, soundfile as sf
from scipy.special import expit
from scipy.signal import resample_poly

from aedes_binary import targets, weights, ROOT, decisions
from aedes_classical import raw_scores, features as extract_base_features
from train_aedes_binary import summary
from bioacoustic_invariance import (
    extract_batch_bioacoustic_features,
    BIOACOUSTIC_FEATURE_NAMES,
    SR
)
from models_committee import StackingMetaCommittee, SoftVotingCommittee

BENCHMARK_DIR = ROOT / 'results/corrected_2026_10_01'
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_predictions(y_true, scores, quality, low, high, domain_name, model_name):
    """Compute complete suite of metrics for a set of scores."""
    states = decisions(scores, quality, low, high)
    df = pd.DataFrame(dict(
        label=np.where(y_true == 1, 'Aedes aegypti', np.where(y_true == 0, 'non_aedes', 'unknown_mosquito')),
        quality_ok=quality,
        aedes_score=scores,
        state=states
    ))
    res, _ = summary(df, scores, low, high)
    return dict(
        model=model_name,
        domain=domain_name,
        total_windows=len(df),
        **res
    )

def run_comprehensive_benchmark():
    """Evaluate frozen corrected models, recording-disjoint noise and separate D5 test."""
    from corrected_validation import evaluate
    return evaluate()

if __name__ == '__main__':
    run_comprehensive_benchmark()
