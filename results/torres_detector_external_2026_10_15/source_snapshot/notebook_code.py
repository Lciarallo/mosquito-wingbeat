# %% Setup
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.fft import dct
import soundfile as sf
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
import joblib
import torch
from torch import nn
from IPython.display import display, Audio, Markdown

ROOT = Path(os.environ.get("MOSQUITO_ROOT", Path.cwd())).resolve()
assert (ROOT / "fetch_data.py").exists(), "Abra o notebook na pasta mosquito-wingbeat."
sys.path.insert(0, str(ROOT))
from fetch_data import fetch_dataset, fetch_noise
DATA, CACHE, RESULTS = ROOT / "data", ROOT / "data/cache", ROOT / "results"
for directory in [CACHE, RESULTS, RESULTS / "models", RESULTS / "figures"]:
    directory.mkdir(parents=True, exist_ok=True)

CONFIG = dict(seed=42, sample_rate=16000, window_s=1.0, max_windows_per_file=60,
              folds=3, mel_bands=40, fft_size=512, hop_samples=256,
              min_frequency_hz=200, max_frequency_hz=900, feature_version="1.0",
              cnn_epochs=25, cnn_patience=5, cnn_batch_size=128)
SEED, SR = CONFIG["seed"], CONFIG["sample_rate"]
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
# Deterministic algorithms are requested; unsupported GPU kernels emit warnings.
torch.use_deterministic_algorithms(True, warn_only=True)
warnings.filterwarnings("once", message=".*adaptive_avg_pool2d_backward_cuda.*")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 180, "axes.spines.top": False,
                     "axes.spines.right": False})

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def save_figure(name):
    plt.savefig(RESULTS / "figures" / f"{name}.png", bbox_inches="tight")
    plt.show()

versions = {name: importlib.metadata.version(name) for name in
            ["numpy", "scipy", "pandas", "matplotlib", "scikit-learn", "soundfile",
             "torch", "requests", "rarfile", "libarchive-c", "nbformat", "nbclient"]}
print("Dispositivo de treinamento:", DEVICE)
print("Python:", sys.version.split()[0], "| Versões:", versions)
display(pd.Series(CONFIG, name="Configuração"))

# %% Download
archives = fetch_dataset(ROOT, workers=3)
noise_provenance = fetch_noise(ROOT)
assert len(archives) == 20
assert all(record["md5"] == record["official_md5"] for record in archives)
print(f"20/20 arquivos íntegros; {sum(r['size'] for r in archives):,} bytes.")
display(pd.DataFrame(archives)[["filename", "size", "md5", "sha256"]])

# %% Inventory functions
AUDIO_EXTENSIONS = {".wav", ".m4a", ".mp4", ".amr", ".mp3", ".flac", ".aiff", ".ogg"}

def infer_phone(path):
    text = str(path).lower()
    patterns = [("T209", r"t209"), ("Nexus", r"nexus"), ("Xperia", r"xperia"),
                ("iPhone4S", r"iphone.?4s"), ("iPhone6", r"iphone.?6"),
                ("iPhone", r"iphone"), ("Nokia5235", r"nokia.?5235"),
                ("Nokia6555B", r"nokia.?6555"), ("Huawei866C", r"huawei|866c"),
                ("HTC", r"htc")]
    return next((label for label, pattern in patterns if re.search(pattern, text)), "unknown")

def noise_source(filename):
    name = filename.lower()
    if "background" in name:
        return "background_unresolved"
    if "26nov21" in name:
        return "authors_26Nov21"
    # The final numeric suffix is the offset of a cut from the same source.
    name = re.sub(r"\.\d+\.wav$", "", name)
    name = re.sub(r"^16khz_\d+_", "", name)
    return name

def audio_info(path):
    try:
        info = sf.info(path)
        return info.samplerate, info.channels, info.frames / info.samplerate, info.subtype
    except Exception:
        result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                                 "-show_entries", "stream=sample_rate,channels,codec_name:format=duration",
                                 "-of", "json", str(path)], capture_output=True, text=True, check=True)
        metadata = json.loads(result.stdout)
        audio = metadata["streams"][0]
        return (int(audio["sample_rate"]), int(audio["channels"]),
                float(metadata["format"]["duration"]), audio["codec_name"])

def build_inventory():
    files = sorted((DATA / "raw").rglob("*")) + sorted((DATA / "noise").rglob("*"))
    files = [p for p in files if p.is_file() and not p.name.startswith(".")
             and "__MACOSX" not in str(p)]
    wav_sources = {(str(p.parent).replace("/Raw", ""), p.stem.lower())
                   for p in files if p.suffix.lower() == ".wav"}
    rows = []
    for path in files:
        source = "tinyml_noise" if path.is_relative_to(DATA / "noise") else "dryad"
        species = "noise" if source == "tinyml_noise" else path.relative_to(DATA / "raw").parts[0]
        explicit_noise = species == "noise" or "background" in path.name.lower()
        label = "noise" if explicit_noise else species
        reason = ""
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            reason = "non_audio"
        elif species == "Aedes sierrensis" and "CleanedData_Snipped" not in str(path):
            reason = "sierrensis_uncurated_or_copy"
        elif species != "Aedes sierrensis" and "snipped" in str(path).lower():
            reason = "derived_excerpt_unmapped"
        elif path.suffix.lower() != ".wav" and (str(path.parent).replace("/Raw", ""), path.stem.lower()) in wav_sources:
            reason = "compressed_original_has_wav"
        digest = sha256_file(path)
        rate = channels = duration = None
        subtype = ""
        if not reason:
            try:
                rate, channels, duration, subtype = audio_info(path)
                if duration < CONFIG["window_s"]:
                    reason = "shorter_than_window"
            except Exception as error:
                reason = "decode_error:" + type(error).__name__
        collector = re.match(r"([A-Za-z]+)", path.stem)
        session = (species + "|field_" + collector.group(1).lower()
                   if species == "Aedes sierrensis" and collector
                   else species + "|" + str(path.relative_to(DATA).parent))
        group = "noise|" + noise_source(path.name) if explicit_noise else digest
        rows.append(dict(path=str(path.relative_to(ROOT)), source=source, species=species,
                         label=label, phone=infer_phone(path), session=session,
                         group=group, sha256=digest, bytes=path.stat().st_size,
                         sample_rate=rate, channels=channels, duration_s=duration,
                         subtype=subtype, extension=path.suffix.lower(), exclusion=reason))
    inventory = pd.DataFrame(rows)
    valid = inventory.exclusion.eq("")
    conflicts = inventory.loc[valid].groupby("sha256").label.nunique()
    ambiguous = set(conflicts[conflicts > 1].index)
    inventory.loc[valid & inventory.sha256.isin(ambiguous), "exclusion"] = "conflicting_exact_labels"
    valid = inventory.exclusion.eq("")
    duplicates = inventory.loc[valid].duplicated("sha256", keep="first")
    inventory.loc[duplicates.index[duplicates], "exclusion"] = "exact_duplicate"
    inventory.to_csv(RESULTS / "inventory.csv", index=False)
    return inventory

def read_wave(row):
    path = ROOT / row["path"]
    try:
        wave, rate = sf.read(path, dtype="float32", always_2d=True)
        wave = wave.mean(axis=1)
        divisor = math.gcd(rate, SR)
        if rate != SR:
            wave = signal.resample_poly(wave, SR // divisor, rate // divisor).astype("float32")
    except Exception:
        decoded = CACHE / "decoded" / f"{row['sha256']}.wav"
        decoded.parent.mkdir(exist_ok=True)
        if not decoded.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(path),
                            "-ac", "1", "-ar", str(SR), "-c:a", "pcm_f32le", str(decoded)],
                           check=True, capture_output=True)
        wave, _ = sf.read(decoded, dtype="float32")
    return np.nan_to_num(wave).astype("float32")

# %% Inventory
inventory = build_inventory()
usable = inventory.loc[inventory.exclusion.eq("")].copy()
display(inventory.exclusion.replace("", "included").value_counts().rename("Arquivos"))
display(usable.groupby("label").agg(recordings=("path", "size"), groups=("group", "nunique"),
                                    minutes=("duration_s", lambda x: x.sum() / 60)))
print("Taxas declaradas nos arquivos:", sorted(usable.sample_rate.dropna().unique()))
print("Formatos inventariados:", inventory.extension.value_counts().to_dict())
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
counts = usable[usable.label != "noise"].groupby("label").size().sort_values()
counts.plot.barh(ax=axes[0], color="#2563eb")
axes[0].set(xlabel="Gravações candidatas", ylabel="", title="Suporte por espécie após auditoria")
usable.groupby("phone").duration_s.sum().div(60).sort_values().plot.barh(ax=axes[1], color="#d97706")
axes[1].set(xlabel="Minutos", ylabel="", title="Aparelhos inferidos das pastas")
plt.tight_layout()
save_figure("inventory")

# %% Feature functions
def mel_filterbank():
    frequencies = np.fft.rfftfreq(CONFIG["fft_size"], 1 / SR)
    to_mel = lambda f: 2595 * np.log10(1 + f / 700)
    to_hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    limits = to_hz(np.linspace(to_mel(100), to_mel(3900), CONFIG["mel_bands"] + 2))
    bank = np.zeros((CONFIG["mel_bands"], len(frequencies)), dtype="float32")
    for k in range(len(bank)):
        bank[k] = np.maximum(0, np.minimum((frequencies - limits[k]) / (limits[k+1] - limits[k]),
                                          (limits[k+2] - frequencies) / (limits[k+2] - limits[k+1])))
        bank[k] /= max(bank[k].sum(), 1e-12)
    return bank

MEL_BANK = mel_filterbank()
PSD_BINS = np.arange(150, 3951, 50)
F0_BINS = np.arange(200, 905, 5)
FEATURE_NAMES = ([f"mfcc_mean_{i}" for i in range(13)] + [f"mfcc_std_{i}" for i in range(13)] +
                 [f"psd_{int(f)}Hz" for f in PSD_BINS] +
                 ["f0_median", "f0_std", "f0_q10", "f0_q90", "peak_frequency",
                  "harmonic_f0", "spectral_centroid", "spectral_flatness", "spectral_entropy"])

def wave_features(raw_wave):
    wave = np.asarray(raw_wave, dtype="float32")
    wave = wave - wave.mean()
    rms = float(np.sqrt(np.mean(wave**2)))
    wave = wave / max(rms, 1e-8)
    freqs, _, spectrum = signal.stft(wave, fs=SR, nperseg=CONFIG["fft_size"],
                                     noverlap=CONFIG["fft_size"] - CONFIG["hop_samples"],
                                     boundary=None, padded=False)
    mel = MEL_BANK @ np.abs(spectrum)**2
    mel_db = 10 * np.log10(np.maximum(mel, 1e-12))
    mel_db = np.clip(mel_db - mel_db.max(), -80, 0)
    mfcc = dct(mel_db, type=2, axis=0, norm="ortho")[:13]
    f, psd = signal.welch(wave, fs=SR, nperseg=3200, noverlap=2880)
    selected = (f >= 150) & (f <= 3900)
    probabilities = psd[selected] / max(psd[selected].sum(), 1e-12)
    log_psd = 10 * np.log10(np.maximum(np.interp(PSD_BINS, f, psd), 1e-12))
    log_psd -= log_psd.max()
    # Long windows: nominal 5-Hz bins and 20-ms step, matching the 2017 design.
    ff, _, long_stft = signal.stft(wave, fs=SR, nperseg=3200, noverlap=2880,
                                   boundary=None, padded=False)
    band = (ff >= 200) & (ff <= 900)
    track = ff[band][np.abs(long_stft[band]).argmax(axis=0)]
    hist = np.histogram(track, bins=F0_BINS)[0].astype("float32")
    hist /= max(hist.sum(), 1)
    candidates = np.arange(200, 901, 5)
    harmonic_scores = sum(np.interp(candidates * harmonic, f, psd) / harmonic
                          for harmonic in [1, 2, 3])
    peak_frequency = f[(f >= 200) & (f <= 900)][psd[(f >= 200) & (f <= 900)].argmax()]
    summary = [np.median(track), np.std(track), *np.quantile(track, [.1, .9]),
               peak_frequency, candidates[harmonic_scores.argmax()],
               np.sum(f[selected] * probabilities),
               np.exp(np.mean(np.log(np.maximum(probabilities, 1e-12)))) / max(probabilities.mean(), 1e-12),
               -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)))]
    features = np.r_[mfcc.mean(axis=1), mfcc.std(axis=1), log_psd, summary].astype("float32")
    return features, hist, (mel_db / 80).astype("float32"), rms

def prepare_features(usable):
    fingerprint = hashlib.sha256((usable[["path", "sha256", "group", "label"]].to_json() +
                                   json.dumps(CONFIG, sort_keys=True)).encode()).hexdigest()
    marker = CACHE / "features_manifest.json"
    if marker.exists() and json.loads(marker.read_text())["fingerprint"] == fingerprint:
        arrays = np.load(CACHE / "features.npz")
        return (pd.read_csv(CACHE / "segments.csv"), arrays["features"], arrays["histograms"],
                np.load(CACHE / "waves.npy", mmap_mode="r"), np.load(CACHE / "mel.npy", mmap_mode="r"))
    segments, features, histograms, waves, spectrograms = [], [], [], [], []
    for number, (_, row) in enumerate(usable.iterrows(), 1):
        wave = read_wave(row)
        n_windows = len(wave) // SR
        positions = np.unique(np.linspace(0, n_windows - 1, min(n_windows, CONFIG["max_windows_per_file"])).astype(int))
        for position in positions:
            window = wave[position * SR:(position + 1) * SR]
            if len(window) != SR or np.sqrt(np.mean(window**2)) < 1e-7:
                continue
            feature, histogram, mel, rms = wave_features(window)
            normalized = window - window.mean()
            normalized /= max(np.sqrt(np.mean(normalized**2)), 1e-8)
            content = hashlib.sha256(np.round(window * 32767).astype("int32").tobytes()).hexdigest()
            segments.append(dict(path=row.path, label=row.label, group=row.group, phone=row.phone,
                                 session=row.session, start_s=int(position), duration_s=1,
                                 rms=rms, content_sha256=content))
            features.append(feature); histograms.append(histogram)
            waves.append(normalized.astype("float32")); spectrograms.append(mel)
        if number % 50 == 0 or number == len(usable):
            print(f"Features: {number}/{len(usable)} gravações, {len(segments)} janelas", flush=True)
    segments = pd.DataFrame(segments)
    # Connect groups sharing an identical waveform; remove duplicated windows.
    parent = {group: group for group in segments.group.unique()}
    def find(group):
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group
    conflicted = set()
    for digest, subset in segments.groupby("content_sha256"):
        groups = subset.group.unique()
        if len(groups) > 1:
            if subset.label.nunique() > 1:
                conflicted.add(digest)
            else:
                for group in groups[1:]:
                    parent[find(group)] = find(groups[0])
    segments["group"] = segments.group.map(find)
    keep = (~segments.content_sha256.isin(conflicted) &
            ~segments.duplicated("content_sha256", keep="first")).to_numpy()
    duplicate_audit = dict(conflicting_waveforms=len(conflicted),
                           dropped_windows=int((~keep).sum()), retained_windows=int(keep.sum()))
    segments = segments.loc[keep].reset_index(drop=True)
    arrays = [np.stack(items)[keep] for items in [features, histograms, waves, spectrograms]]
    np.savez_compressed(CACHE / "features.npz", features=arrays[0], histograms=arrays[1])
    np.save(CACHE / "waves.npy", arrays[2]); np.save(CACHE / "mel.npy", arrays[3])
    segments.to_csv(CACHE / "segments.csv", index=False)
    marker.write_text(json.dumps(dict(fingerprint=fingerprint, config=CONFIG,
                                     segment_audit=duplicate_audit), indent=2))
    return segments, arrays[0], arrays[1], arrays[2], arrays[3]

# %% Features
segments, X, H, WAVES, MEL = prepare_features(usable)
assert X.shape[1] == len(FEATURE_NAMES)
assert len(segments) == len(X) == len(H) == len(WAVES) == len(MEL)
assert np.isfinite(X).all() and np.isfinite(H).all()
segments.to_csv(RESULTS / "segments.csv", index=False)
support = segments.groupby("label").agg(windows=("label", "size"), groups=("group", "nunique"))
display(support)
display(json.loads((CACHE / "features_manifest.json").read_text())["segment_audit"])
examples = {}
for label in ["Aedes aegypti", "Aedes albopictus", "Anopheles gambiae", "Culex pipiens", "noise"]:
    candidates = segments.index[segments.label.eq(label)]
    if len(candidates):
        # An example is selected for illustration; it is not an activity annotation.
        examples[label] = candidates[len(candidates) // 2]
fig, axes = plt.subplots(len(examples), 2, figsize=(13, 2.5 * len(examples)), squeeze=False)
for (label, idx), pair in zip(examples.items(), axes):
    f, psd = signal.welch(WAVES[idx], SR, nperseg=3200)
    pair[0].plot(f, 10 * np.log10(np.maximum(psd, 1e-12)))
    pair[0].set(xlim=(100, 3900), xlabel="Frequência (Hz)", ylabel="PSD (dB)", title=label)
    pair[1].imshow(MEL[idx], aspect="auto", origin="lower", extent=[0, 1, 0, 40], cmap="magma", vmin=-1, vmax=0)
    pair[1].set(xlabel="Tempo (s)", ylabel="Banda Mel", title="Espectrograma relativo, 100–3900 Hz")
plt.tight_layout()
save_figure("signal_examples")
for label in ["Aedes aegypti", "Aedes albopictus", "noise"]:
    if label in examples:
        idx = examples[label]
        print(label, "|", segments.iloc[idx]["path"], "| início:", segments.iloc[idx].start_s)
        display(Audio(np.asarray(WAVES[idx]), rate=SR))

# %% Validation functions
def assign_folds(labels, groups):
    group_table = pd.DataFrame({"group": groups, "label": labels})
    assert group_table.groupby("group").label.nunique().max() == 1
    table = group_table.drop_duplicates("group").reset_index(drop=True)
    support = table.label.value_counts()
    included = table[table.label.isin(support[support >= CONFIG["folds"]].index)].copy()
    splitter = StratifiedKFold(n_splits=CONFIG["folds"], shuffle=True, random_state=SEED)
    assignments = {}
    for fold, (_, test) in enumerate(splitter.split(included, included.label)):
        assignments.update({group: fold for group in included.iloc[test].group})
    return np.array([assignments.get(group, -1) for group in groups]), support

def nested_holdout(indices, labels):
    rng = np.random.default_rng(SEED + len(indices))
    table = pd.DataFrame({"group": segments.iloc[indices].group.to_numpy(),
                          "label": labels[indices]}).drop_duplicates("group")
    validation_groups = []
    for label, subset in table.groupby("label"):
        groups = subset.group.to_numpy().copy()
        if len(groups) < 2:
            raise ValueError(f"Faltam grupos para treino/validação: {label}")
        rng.shuffle(groups)
        validation_groups.extend(groups[:max(1, int(np.ceil(len(groups) * .2)))])
    is_validation = segments.iloc[indices].group.isin(validation_groups).to_numpy()
    train, validation = indices[~is_validation], indices[is_validation]
    assert set(segments.iloc[train].group).isdisjoint(segments.iloc[validation].group)
    return train, validation

def group_weights(indices, labels):
    frame = pd.DataFrame({"group": segments.iloc[indices].group.to_numpy(),
                          "label": labels[indices]})
    group_sizes = frame.groupby("group").size()
    class_groups = frame.drop_duplicates("group").groupby("label").size()
    weights = 1 / (frame.group.map(group_sizes).to_numpy() * frame.label.map(class_groups).to_numpy())
    return weights / weights.mean()

def as_probabilities(model, features):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)
    scores = model.decision_function(features)
    if scores.ndim == 1:
        scores = np.c_[-scores, scores]
    scores = scores - scores.max(axis=1, keepdims=True)
    return np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)

def metrics(y_true, predicted):
    return dict(accuracy=accuracy_score(y_true, predicted),
                balanced_accuracy=balanced_accuracy_score(y_true, predicted),
                macro_f1=f1_score(y_true, predicted, average="macro", zero_division=0))

def aggregate_recordings(indices, y_true, probabilities, class_names):
    frame = segments.iloc[indices][["path", "group", "phone", "session"]].copy()
    frame["true"] = y_true
    probability_columns = [f"p{i}" for i in range(len(class_names))]
    frame[probability_columns] = probabilities
    result = frame.groupby("group").agg(path=("path", "first"), phone=("phone", "first"),
                                       session=("session", "first"), true=("true", "first"),
                                       windows=("true", "size"))
    scores = frame.groupby("group")[probability_columns].mean().to_numpy()
    result["predicted"] = scores.argmax(axis=1)
    result["true_label"] = np.asarray(class_names)[result.true.to_numpy().astype(int)]
    result["predicted_label"] = np.asarray(class_names)[result.predicted.to_numpy().astype(int)]
    result["max_score"] = scores.max(axis=1)
    return result.reset_index()

def bootstrap_macro_recall(recordings, repeats=1000):
    rng = np.random.default_rng(SEED)
    # Resample recording/source groups within each species, not audio windows.
    classes = [(row.true == row.predicted).to_numpy() for _, row in recordings.groupby("true")]
    values = []
    for _ in range(repeats):
        recalls = []
        for correct in classes:
            recalls.append(correct[rng.integers(0, len(correct), len(correct))].mean())
        values.append(np.mean(recalls))
    return np.quantile(values, [.025, .975])

class FrequencyDistributionClassifier:
    """Additive density score (2017 description) or smoothed log likelihood."""
    def __init__(self, logarithmic=False):
        self.logarithmic = logarithmic

    def fit(self, histograms, labels, sample_weight=None):
        self.classes_ = np.unique(labels)
        weights = np.ones(len(labels)) if sample_weight is None else sample_weight
        self.references_ = np.stack([np.average(histograms[labels == label], axis=0,
                                               weights=weights[labels == label])
                                     for label in self.classes_])
        # Fixed Dirichlet-style smoothing: finite logs for unseen frequency bins.
        self.references_ = self.references_ + 1e-4
        self.references_ /= self.references_.sum(axis=1, keepdims=True)
        return self

    def predict_proba(self, histograms):
        references = np.log(self.references_) if self.logarithmic else self.references_
        scores = histograms @ references.T
        scores -= scores.max(axis=1, keepdims=True)
        return np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)

# %% Splits
fold_ids, group_support = assign_folds(segments.label.to_numpy(), segments.group.to_numpy())
segments["fold_species"] = fold_ids
species_encoder = LabelEncoder().fit(segments.loc[(fold_ids >= 0) & segments.label.ne("noise"), "label"])
SPECIES = species_encoder.classes_
Y_SPECIES = np.full(len(segments), -1, dtype=int)
species_mask = segments.label.isin(SPECIES).to_numpy()
Y_SPECIES[species_mask] = species_encoder.transform(segments.loc[species_mask, "label"])
PRIMARY = np.flatnonzero((fold_ids >= 0) & (Y_SPECIES >= 0))
TINY_NAMES = np.array(["aegypti", "albopictus", "noise", "other"])
tiny_labels = segments.label.map(lambda label: {"Aedes aegypti": "aegypti", "Aedes albopictus": "albopictus",
                                               "noise": "noise"}.get(label, "other")).to_numpy()
Y_TINY = np.searchsorted(TINY_NAMES, tiny_labels)
tiny_folds, _ = assign_folds(tiny_labels, segments.group.to_numpy())
segments["fold_tinyml"] = tiny_folds
segments.to_csv(RESULTS / "segments.csv", index=False)
for assignments in [fold_ids, tiny_folds]:
    assert pd.DataFrame({"group": segments.group, "fold": assignments}).groupby("group").fold.nunique().max() == 1
for fold in range(CONFIG["folds"]):
    train = PRIMARY[fold_ids[PRIMARY] != fold]
    test = PRIMARY[fold_ids[PRIMARY] == fold]
    assert set(segments.iloc[train].group).isdisjoint(segments.iloc[test].group)
    assert set(segments.iloc[train].content_sha256).isdisjoint(segments.iloc[test].content_sha256)
    assert len(np.unique(Y_SPECIES[train])) == len(SPECIES)
    assert len(np.unique(Y_SPECIES[test])) == len(SPECIES)
display(pd.crosstab(segments.loc[PRIMARY, "label"], fold_ids[PRIMARY]))
print("Espécies efetivamente avaliadas:", len(SPECIES))
print("Espécies sem grupos suficientes:", group_support[group_support < CONFIG["folds"]].to_dict())
print("Janelas no benchmark:", len(PRIMARY), "| grupos:", segments.iloc[PRIMARY].group.nunique())

# %% Classical models
MFCC_COLUMNS = np.arange(26)
PSD_COLUMNS = np.arange(26, 26 + len(PSD_BINS))
MODEL_SPECS = {
    "Majoritária": (DummyClassifier(strategy="most_frequent"), X, np.arange(X.shape[1])),
    "F0: soma de densidades": (FrequencyDistributionClassifier(False), H, None),
    "F0: log-verossimilhança": (FrequencyDistributionClassifier(True), H, None),
    "PSD + SVM linear": (make_pipeline(StandardScaler(), LinearSVC(C=1, max_iter=6000, random_state=SEED)),
                         X[:, PSD_COLUMNS], PSD_COLUMNS),
    "MFCC + regressão logística": (make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000, random_state=SEED)),
                                    X[:, MFCC_COLUMNS], MFCC_COLUMNS),
    "MFCC/PSD/F0 + ExtraTrees": (ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2,
                                                     max_features="sqrt", n_jobs=4, random_state=SEED),
                                 X, np.arange(X.shape[1])),
}
CLASSICAL_MODELS, OOF, RECORDINGS, summary_rows = {}, {}, {}, []
for name, (prototype, features, columns) in MODEL_SPECS.items():
    probabilities = np.zeros((len(PRIMARY), len(SPECIES)), dtype="float32")
    trained = []
    elapsed = 0
    for fold in range(CONFIG["folds"]):
        train = PRIMARY[fold_ids[PRIMARY] != fold]
        locations = np.flatnonzero(fold_ids[PRIMARY] == fold)
        test = PRIMARY[locations]
        model = copy.deepcopy(prototype)
        weights = group_weights(train, Y_SPECIES)
        begin = time.perf_counter()
        if isinstance(model, FrequencyDistributionClassifier):
            model.fit(features[train], Y_SPECIES[train], sample_weight=weights)
        elif isinstance(model, DummyClassifier):
            model.fit(features[train], Y_SPECIES[train])
        elif hasattr(model, "steps"):
            model.fit(features[train], Y_SPECIES[train],
                      **{model.steps[-1][0] + "__sample_weight": weights})
        else:
            model.fit(features[train], Y_SPECIES[train], sample_weight=weights)
        elapsed += time.perf_counter() - begin
        probabilities[locations] = as_probabilities(model, features[test])
        trained.append(model)
    CLASSICAL_MODELS[name] = trained
    OOF[name] = probabilities
    recording_predictions = aggregate_recordings(PRIMARY, Y_SPECIES[PRIMARY], probabilities, SPECIES)
    RECORDINGS[name] = recording_predictions
    low, high = bootstrap_macro_recall(recording_predictions)
    row = dict(model=name, task="species", n_classes=len(SPECIES), n_windows=len(PRIMARY),
               n_groups=len(recording_predictions), **metrics(Y_SPECIES[PRIMARY], probabilities.argmax(axis=1)),
               recording_macro_recall=balanced_accuracy_score(recording_predictions.true, recording_predictions.predicted),
               ci_low=low, ci_high=high, training_seconds=elapsed)
    summary_rows.append(row)
    print(name, "| acurácia balanceada por grupo:", round(row["recording_macro_recall"], 3), flush=True)
    recording_predictions.to_csv(RESULTS / f"predictions_classical_{len(summary_rows)}.csv", index=False)
joblib.dump(dict(models=CLASSICAL_MODELS, classes=SPECIES, config=CONFIG),
            RESULTS / "models/classical_folds.joblib", compress=3)
classical_summary = pd.DataFrame(summary_rows)
classical_summary.to_csv(RESULTS / "classical_metrics.csv", index=False)
display(classical_summary.round(4))

# %% Acoustic overlap
reference_histograms = []
for label in SPECIES:
    indices = np.flatnonzero(segments.label.eq(label).to_numpy())
    weights = group_weights(indices, Y_SPECIES)
    histogram = np.average(H[indices], axis=0, weights=weights) + 1e-6
    reference_histograms.append(histogram / histogram.sum())
reference_histograms = np.stack(reference_histograms)
bhattacharyya = np.sqrt(reference_histograms) @ np.sqrt(reference_histograms).T
fig, axes = plt.subplots(1, 2, figsize=(17, 7))
sns.heatmap(bhattacharyya, vmin=0, vmax=1, cmap="viridis", xticklabels=SPECIES,
            yticklabels=SPECIES, ax=axes[0], cbar_kws={"label": "BC"})
axes[0].set_title("Sobreposição acústica descritiva: todas as gravações")
name = "MFCC/PSD/F0 + ExtraTrees"
matrix = confusion_matrix(Y_SPECIES[PRIMARY], OOF[name].argmax(axis=1), normalize="true")
sns.heatmap(matrix, vmin=0, vmax=1, cmap="Blues", xticklabels=SPECIES, yticklabels=SPECIES,
            ax=axes[1], cbar_kws={"label": "Fração de janelas"})
axes[1].set(title="ExtraTrees: previsões fora da dobra", xlabel="Predição", ylabel="Espécie real")
plt.tight_layout()
save_figure("acoustic_overlap_and_confusion")

# %% CNN functions
class AudioFrontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(CONFIG["fft_size"]))
        self.register_buffer("mel_bank", torch.tensor(MEL_BANK))

    def forward(self, wave):
        wave = wave - wave.mean(dim=-1, keepdim=True)
        wave = wave / wave.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
        spectrum = torch.stft(wave, n_fft=CONFIG["fft_size"], hop_length=CONFIG["hop_samples"],
                              window=self.window, center=False, return_complex=True) / self.window.sum()
        mel = self.mel_bank @ spectrum.abs().square()
        decibels = 10 * torch.log10(mel.clamp_min(1e-12))
        decibels = decibels - decibels.amax(dim=(-2, -1), keepdim=True)
        return decibels.clamp(-80, 0) / 80

class AudioCNN(nn.Module):
    def __init__(self, classes, dimensions=1):
        super().__init__()
        self.frontend = AudioFrontend()
        self.dimensions = dimensions
        if dimensions == 1:
            self.classifier = nn.Sequential(
                nn.Conv1d(40, 24, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(24, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(.5), nn.Linear(32, classes))
        else:
            self.classifier = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
                nn.Linear(24 * 16, 96), nn.ReLU(), nn.Dropout(.3), nn.Linear(96, classes))

    def forward(self, wave):
        features = self.frontend(wave)
        return self.classifier(features if self.dimensions == 1 else features.unsqueeze(1))

def cnn_predict(model, waves, batch_size=128):
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(waves), batch_size):
            batch = torch.from_numpy(np.array(waves[start:start + batch_size], dtype="float32", copy=True)).to(DEVICE)
            output.append(model(batch).softmax(dim=1).cpu().numpy())
    return np.concatenate(output)

def train_cnn(outer_train, outer_test, labels, class_names, dimensions, augment_noise, tag, outer_fold):
    train, validation = nested_holdout(outer_train, labels)
    assert set(segments.iloc[validation].group).isdisjoint(segments.iloc[outer_test].group)
    assert set(segments.iloc[train].group).isdisjoint(segments.iloc[outer_test].group)
    if "noise" in class_names:
        noise_train = train[segments.iloc[train].label.eq("noise").to_numpy()]
    else:
        noise_train = np.flatnonzero(segments.label.eq("noise").to_numpy() &
                                     (fold_ids >= 0) & (fold_ids != outer_fold))
    assert set(segments.iloc[noise_train].group).isdisjoint(segments.iloc[outer_test].group)
    signature = hashlib.sha256((json.loads((CACHE / "features_manifest.json").read_text())["fingerprint"] +
                                json.dumps(dict(train=train.tolist(), validation=validation.tolist(),
                                                dimensions=dimensions, noise=augment_noise, tag=tag))).encode()).hexdigest()
    checkpoint = RESULTS / "models" / f"{tag}_fold{outer_fold}.pt"
    torch.manual_seed(SEED + outer_fold)
    model = AudioCNN(len(class_names), dimensions).to(DEVICE)
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["signature"] == signature:
            model.load_state_dict(saved["state_dict"])
            print(f"{tag}, dobra {outer_fold}: checkpoint validado e reutilizado", flush=True)
            return model, saved["history"]
    train_wave = torch.from_numpy(np.array(WAVES[train], copy=True))
    train_labels = torch.tensor(labels[train], dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(train_wave, train_labels)
    generator = torch.Generator().manual_seed(SEED + outer_fold)
    sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(group_weights(train, labels), dtype=torch.double),
                                                     len(train), replacement=True, generator=generator)
    loader = torch.utils.data.DataLoader(dataset, batch_size=CONFIG["cnn_batch_size"], sampler=sampler,
                                        num_workers=0, pin_memory=DEVICE.type == "cuda")
    noise_pool = torch.from_numpy(np.array(WAVES[noise_train], copy=True)).to(DEVICE) if len(noise_train) else None
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    history, best_state, best_loss, patience = [], None, np.inf, 0
    validation_weights = group_weights(validation, labels)
    begin = time.perf_counter()
    for epoch in range(CONFIG["cnn_epochs"]):
        model.train()
        running_loss = 0
        for wave, target in loader:
            wave, target = wave.to(DEVICE), target.to(DEVICE)
            if augment_noise and noise_pool is not None:
                # Only training noise sources are used; test backgrounds remain unseen.
                noise = noise_pool[torch.randint(len(noise_pool), (len(wave),), device=DEVICE)]
                snr_db = torch.empty((len(wave), 1), device=DEVICE).uniform_(5, 25)
                apply = (torch.rand((len(wave), 1), device=DEVICE) < .5).float()
                wave = wave + noise * (10 ** (-snr_db / 20)) * apply
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(wave), target)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach()) * len(wave)
        validation_probabilities = cnn_predict(model, WAVES[validation])
        losses = -np.log(np.maximum(validation_probabilities[np.arange(len(validation)), labels[validation]], 1e-9))
        validation_loss = float(np.average(losses, weights=validation_weights))
        row = dict(epoch=epoch + 1, train_loss=running_loss / len(train),
                   validation_loss=validation_loss,
                   validation_balanced_accuracy=balanced_accuracy_score(labels[validation], validation_probabilities.argmax(1)))
        history.append(row)
        if validation_loss < best_loss - 1e-4:
            best_loss, patience = validation_loss, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience += 1
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"{tag}, dobra {outer_fold}, época {epoch + 1}: loss val={validation_loss:.3f}", flush=True)
        if patience >= CONFIG["cnn_patience"]:
            break
    model.load_state_dict(best_state)
    torch.save(dict(state_dict=best_state, signature=signature, history=history,
                    config=CONFIG, classes=class_names.tolist(), dimensions=dimensions,
                    train_groups=segments.iloc[train].group.unique().tolist(),
                    validation_groups=segments.iloc[validation].group.unique().tolist(),
                    test_groups=segments.iloc[outer_test].group.unique().tolist(),
                    training_seconds=time.perf_counter() - begin), checkpoint)
    return model, history

# %% CNN benchmark
frontend = AudioFrontend().to(DEVICE)
with torch.no_grad():
    check = frontend(torch.from_numpy(np.array(WAVES[:8], copy=True)).to(DEVICE)).cpu().numpy()
frontend_error = float(np.max(np.abs(check - MEL[:8])))
assert frontend_error < 1e-4, f"Frontends SciPy/Torch divergem: {frontend_error}"
print("Diferença máxima entre frontends:", frontend_error)
CNN_MODELS, CNN_HISTORIES = {}, {}
for name, dimensions, augmented, tag in [
    ("CNN1D compacta", 1, False, "species_cnn1d"),
    ("CNN2D + ruído real no treino", 2, True, "species_cnn2d_noise"),
]:
    probabilities = np.zeros((len(PRIMARY), len(SPECIES)), dtype="float32")
    trained, histories = [], []
    for fold in range(CONFIG["folds"]):
        train = PRIMARY[fold_ids[PRIMARY] != fold]
        locations = np.flatnonzero(fold_ids[PRIMARY] == fold)
        test = PRIMARY[locations]
        model, history = train_cnn(train, test, Y_SPECIES, SPECIES, dimensions, augmented, tag, fold)
        probabilities[locations] = cnn_predict(model, WAVES[test])
        trained.append(model); histories.append(history)
    CNN_MODELS[name], CNN_HISTORIES[name] = trained, histories
    OOF[name] = probabilities
    predictions = aggregate_recordings(PRIMARY, Y_SPECIES[PRIMARY], probabilities, SPECIES)
    RECORDINGS[name] = predictions
    predictions.to_csv(RESULTS / f"predictions_{tag}.csv", index=False)
    low, high = bootstrap_macro_recall(predictions)
    summary_rows.append(dict(model=name, task="species", n_classes=len(SPECIES), n_windows=len(PRIMARY),
                             n_groups=len(predictions), **metrics(Y_SPECIES[PRIMARY], probabilities.argmax(axis=1)),
                             recording_macro_recall=balanced_accuracy_score(predictions.true, predictions.predicted),
                             ci_low=low, ci_high=high,
                             training_seconds=sum(torch.load(RESULTS / "models" / f"{tag}_fold{k}.pt",
                                                              map_location="cpu", weights_only=True)["training_seconds"]
                                                  for k in range(CONFIG["folds"]))))
benchmark = pd.DataFrame(summary_rows)
benchmark.to_csv(RESULTS / "species_metrics.csv", index=False)
np.savez_compressed(RESULTS / "species_oof_probabilities.npz",
                    **{f"model_{k}": probabilities for k, probabilities in enumerate(OOF.values())})
display(benchmark.round(4))
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
order = benchmark.sort_values("recording_macro_recall")
positions = np.arange(len(order))
axes[0].barh(positions, order.recording_macro_recall, color="#2563eb")
axes[0].errorbar(order.recording_macro_recall, positions,
                 xerr=np.stack([order.recording_macro_recall-order.ci_low,
                                order.ci_high-order.recording_macro_recall]).clip(0),
                 fmt="none", color="black", capsize=3)
axes[0].set(yticks=positions, yticklabels=order.model, xlim=(0, 1),
            xlabel="Macro recall por grupo", title="Avaliação fora da dobra; IC descritivo por bootstrap")
for name, histories in CNN_HISTORIES.items():
    for fold, history in enumerate(histories):
        frame = pd.DataFrame(history)
        axes[1].plot(frame.epoch, frame.validation_loss, label=f"{name}, dobra {fold}")
axes[1].set(xlabel="Época", ylabel="Loss de validação", title="Seleção por validação interna")
axes[1].legend(fontsize=7)
plt.tight_layout()
save_figure("species_benchmark")

# %% Four-class TinyML task
TINY = np.flatnonzero(tiny_folds >= 0)
tiny_probabilities = {}
tiny_models = {}
tiny_rows = []
for name, neural in [("MFCC + logística (4 classes)", False), ("CNN1D compacta (4 classes)", True)]:
    probabilities = np.zeros((len(TINY), 4), dtype="float32")
    models = []
    for fold in range(CONFIG["folds"]):
        train = TINY[tiny_folds[TINY] != fold]
        locations = np.flatnonzero(tiny_folds[TINY] == fold)
        test = TINY[locations]
        assert set(segments.iloc[train].group).isdisjoint(segments.iloc[test].group)
        if neural:
            model, _ = train_cnn(train, test, Y_TINY, TINY_NAMES, 1, True, "tinyml_cnn1d", fold)
            probabilities[locations] = cnn_predict(model, WAVES[test])
        else:
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000, random_state=SEED))
            model.fit(X[train][:, MFCC_COLUMNS], Y_TINY[train],
                      logisticregression__sample_weight=group_weights(train, Y_TINY))
            probabilities[locations] = model.predict_proba(X[test][:, MFCC_COLUMNS])
        models.append(model)
    tiny_models[name] = models
    tiny_probabilities[name] = probabilities
    recordings = aggregate_recordings(TINY, Y_TINY[TINY], probabilities, TINY_NAMES)
    recordings.to_csv(RESULTS / f"tinyml_predictions_{int(neural)}.csv", index=False)
    low, high = bootstrap_macro_recall(recordings)
    tiny_rows.append(dict(model=name, n_classes=4, n_windows=len(TINY), n_groups=len(recordings),
                          **metrics(Y_TINY[TINY], probabilities.argmax(1)),
                          recording_macro_recall=balanced_accuracy_score(recordings.true, recordings.predicted),
                          ci_low=low, ci_high=high))
tiny_summary = pd.DataFrame(tiny_rows)
tiny_summary.to_csv(RESULTS / "tinyml_metrics.csv", index=False)
display(tiny_summary.round(4))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for (name, probabilities), ax in zip(tiny_probabilities.items(), axes):
    matrix = confusion_matrix(Y_TINY[TINY], probabilities.argmax(1), labels=np.arange(4), normalize="true")
    sns.heatmap(matrix, vmin=0, vmax=1, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=TINY_NAMES, yticklabels=TINY_NAMES, ax=ax)
    ax.set(title=name, xlabel="Predição", ylabel="Classe real")
plt.tight_layout()
save_figure("tinyml_confusion")

# %% Device transfer
transfer_rows, transfer_predictions = [], []
all_species = np.flatnonzero(Y_SPECIES >= 0)
phones = sorted(set(segments.iloc[all_species].phone) - {"unknown"})
for phone in phones:
    test_candidate = all_species[segments.iloc[all_species].phone.eq(phone).to_numpy()]
    train_candidate = all_species[segments.iloc[all_species].phone.ne(phone).to_numpy()]
    test_labels = set(Y_SPECIES[test_candidate])
    support_train = segments.iloc[train_candidate].assign(target=Y_SPECIES[train_candidate]).groupby("target").group.nunique()
    common = sorted(test_labels & set(support_train[support_train >= 2].index))
    if len(common) < 2:
        continue
    test = test_candidate[np.isin(Y_SPECIES[test_candidate], common)]
    test_groups = set(segments.iloc[test].group)
    train = train_candidate[np.isin(Y_SPECIES[train_candidate], common) &
                            ~segments.iloc[train_candidate].group.isin(test_groups).to_numpy()]
    assert set(segments.iloc[train].group).isdisjoint(test_groups)
    assert len(set(Y_SPECIES[train])) == len(common)
    model = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2, max_features="sqrt",
                                  n_jobs=4, random_state=SEED)
    model.fit(X[train], Y_SPECIES[train], sample_weight=group_weights(train, Y_SPECIES))
    raw = model.predict_proba(X[test])
    full = np.zeros((len(test), len(SPECIES)))
    full[:, model.classes_] = raw
    recordings = aggregate_recordings(test, Y_SPECIES[test], full, SPECIES)
    recordings["held_out_phone"] = phone
    transfer_predictions.append(recordings)
    transfer_rows.append(dict(phone=phone, n_classes=len(common), n_windows=len(test),
                              n_groups=len(recordings), **metrics(Y_SPECIES[test], full.argmax(1)),
                              recording_macro_recall=balanced_accuracy_score(recordings.true, recordings.predicted)))
transfer_summary = pd.DataFrame(transfer_rows)
transfer_summary.to_csv(RESULTS / "device_transfer_metrics.csv", index=False)
if transfer_predictions:
    pd.concat(transfer_predictions).to_csv(RESULTS / "device_transfer_predictions.csv", index=False)
display(transfer_summary.round(4))

# %% Noise robustness
ROBUST_CLASSIC = "MFCC/PSD/F0 + ExtraTrees"
ROBUST_CNN = "CNN2D + ruído real no treino"
robustness_rows = []
for fold in range(CONFIG["folds"]):
    rng = np.random.default_rng(SEED + fold)
    test = PRIMARY[fold_ids[PRIMARY] == fold]
    # Small pre-specified, balanced probe; selection does not inspect predictions.
    chosen = np.concatenate([rng.choice(test[Y_SPECIES[test] == label],
                                        min(8, np.sum(Y_SPECIES[test] == label)), replace=False)
                             for label in range(len(SPECIES))])
    noise_test = np.flatnonzero(segments.label.eq("noise").to_numpy() & (fold_ids == fold))
    assert len(noise_test)
    outer_train = PRIMARY[fold_ids[PRIMARY] != fold]
    assert set(segments.iloc[noise_test].group).isdisjoint(segments.iloc[outer_train].group)
    environmental_noise = np.asarray(WAVES[rng.choice(noise_test, len(chosen), replace=True)])
    gaussian_noise = rng.normal(size=(len(chosen), SR)).astype("float32")
    gaussian_noise /= np.sqrt(np.mean(gaussian_noise**2, axis=1, keepdims=True))
    for kind, background in [("ruído real reservado", environmental_noise), ("ruído gaussiano", gaussian_noise)]:
        for snr in [20, 10, 0]:
            mixed = (np.asarray(WAVES[chosen]) + background * 10 ** (-snr / 20)).astype("float32")
            noisy_features = np.stack([wave_features(wave)[0] for wave in mixed])
            classic = CLASSICAL_MODELS[ROBUST_CLASSIC][fold].predict(noisy_features)
            neural = cnn_predict(CNN_MODELS[ROBUST_CNN][fold], mixed).argmax(1)
            for name, predicted in [(ROBUST_CLASSIC, classic), (ROBUST_CNN, neural)]:
                robustness_rows.append(dict(model=name, fold=fold, noise=kind, added_snr_db=snr,
                                            n_windows=len(chosen), **metrics(Y_SPECIES[chosen], predicted)))
robustness = pd.DataFrame(robustness_rows)
robustness.to_csv(RESULTS / "noise_robustness.csv", index=False)
display(robustness.groupby(["model", "noise", "added_snr_db"])[["balanced_accuracy", "macro_f1"]].mean().round(4))
fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
for (kind, rows), ax in zip(robustness.groupby("noise"), axes):
    sns.lineplot(data=rows, x="added_snr_db", y="balanced_accuracy", hue="model",
                 marker="o", errorbar=None, ax=ax)
    ax.set(title=kind, xlabel="SNR do ruído adicionado (dB)", ylabel="Acurácia balanceada", ylim=(0, 1))
plt.tight_layout()
save_figure("noise_robustness")

# %% Selective classification and metadata diagnostic
probabilities = OOF["MFCC + regressão logística"]
max_scores = probabilities.max(axis=1)
selective_rows = []
for threshold in [0, .3, .5, .7, .9]:
    accepted = max_scores >= threshold
    selective_rows.append(dict(threshold=threshold, coverage=accepted.mean(),
                               accepted_windows=int(accepted.sum()),
                               accepted_accuracy=accuracy_score(Y_SPECIES[PRIMARY][accepted],
                                                                 probabilities[accepted].argmax(1)) if accepted.any() else np.nan))
selective = pd.DataFrame(selective_rows)
selective.to_csv(RESULTS / "selective_classification.csv", index=False)
display(selective.round(4))
# This baseline uses ONLY sampling metadata to expose acquisition confounding.
metadata_frame = segments[["phone"]].copy()
metadata_frame["header_rate"] = segments.path.map(inventory.set_index("path").sample_rate).astype(str)
metadata_predictions = np.full(len(PRIMARY), -1)
for fold in range(CONFIG["folds"]):
    train = PRIMARY[fold_ids[PRIMARY] != fold]
    positions = np.flatnonzero(fold_ids[PRIMARY] == fold)
    test = PRIMARY[positions]
    metadata_train = pd.get_dummies(metadata_frame.iloc[train], dtype=float)
    metadata_test = pd.get_dummies(metadata_frame.iloc[test], dtype=float).reindex(columns=metadata_train.columns, fill_value=0)
    model = LogisticRegression(C=1, max_iter=1000, random_state=SEED)
    model.fit(metadata_train, Y_SPECIES[train], sample_weight=group_weights(train, Y_SPECIES))
    metadata_predictions[positions] = model.predict(metadata_test)
metadata_diagnostic = metrics(Y_SPECIES[PRIMARY], metadata_predictions)
(RESULTS / "metadata_diagnostic.json").write_text(json.dumps(metadata_diagnostic, indent=2))
print("Diagnóstico de confundimento: desempenho sem usar áudio", metadata_diagnostic)
# Only phone and header sampling rate are used; folder species names are excluded.

# %% INT8 functions
def quantize_logistic(model, calibration):
    scaler, classifier = model.steps[0][1], model.steps[-1][1]
    mean = scaler.mean_.astype("float32")
    inverse_std = (1 / scaler.scale_).astype("float32")
    normalized = (calibration.astype("float32") - mean) * inverse_std
    input_scale = np.array(max(np.quantile(np.abs(normalized), .999) / 127, 1e-8), dtype="float32")
    weights = classifier.coef_.astype("float32")
    weight_scale = np.maximum(np.max(np.abs(weights), axis=1) / 127, 1e-8).astype("float32")
    return dict(mean=mean, inverse_std=inverse_std, input_scale=input_scale,
                weights=np.clip(np.rint(weights / weight_scale[:, None]), -127, 127).astype("int8"),
                weight_scale=weight_scale, bias=classifier.intercept_.astype("float32"))

def quantized_predict(payload, features, return_scores=False):
    normalized = (features.astype("float32") - payload["mean"]) * payload["inverse_std"]
    integer_input = np.clip(np.rint(normalized / payload["input_scale"]), -127, 127).astype("int8")
    accumulators = integer_input.astype("int32") @ payload["weights"].astype("int32").T
    scores = accumulators.astype("float32") * payload["input_scale"] * payload["weight_scale"] + payload["bias"]
    return scores if return_scores else scores.argmax(axis=1)

def c_float(value):
    text = format(float(value), ".9g")
    if "." not in text and "e" not in text:
        text += ".0"
    return text + "f"

def export_c(payload):
    count_classes, count_features = payload["weights"].shape
    parts = ["/* MFCC INT8 classifier, fold 0. Audio/MFCC frontend is required separately. */",
             "#ifndef MOSQUITO_MFCC_INT8_H", "#define MOSQUITO_MFCC_INT8_H",
             "#include <stdint.h>", "#include <math.h>",
             f"#define MOSQUITO_CLASSES {count_classes}", f"#define MOSQUITO_FEATURES {count_features}"]
    for name in ["mean", "inverse_std", "weight_scale", "bias"]:
        values = ", ".join(c_float(x) for x in payload[name])
        parts.append(f"static const float mosquito_{name}[{len(payload[name])}] = {{{values}}};")
    parts.append(f"static const float mosquito_input_scale = {c_float(payload['input_scale'])};")
    rows = ",\n".join("{" + ", ".join(str(int(x)) for x in row) + "}" for row in payload["weights"])
    parts.append(f"static const int8_t mosquito_weights[MOSQUITO_CLASSES][MOSQUITO_FEATURES] = {{\n{rows}\n}};")
    parts.append("""
static int mosquito_predict_mfcc(const float input[MOSQUITO_FEATURES]) {
    int8_t quantized[MOSQUITO_FEATURES];
    for (int j = 0; j < MOSQUITO_FEATURES; ++j) {
        float z = (input[j] - mosquito_mean[j]) * mosquito_inverse_std[j] / mosquito_input_scale;
        long value = lrintf(z);
        if (value > 127) value = 127;
        if (value < -127) value = -127;
        quantized[j] = (int8_t)value;
    }
    int best = 0;
    float best_score = -INFINITY;
    for (int k = 0; k < MOSQUITO_CLASSES; ++k) {
        int32_t accumulator = 0;
        for (int j = 0; j < MOSQUITO_FEATURES; ++j)
            accumulator += (int32_t)quantized[j] * mosquito_weights[k][j];
        float score = (float)accumulator * mosquito_input_scale * mosquito_weight_scale[k] + mosquito_bias[k];
        if (score > best_score) { best_score = score; best = k; }
    }
    return best;
}
#endif
""")
    header = RESULTS / "models/mosquito_mfcc_int8.h"
    header.write_text("\n".join(parts))
    demo = RESULTS / "models/mosquito_mfcc_int8_demo.c"
    demo.write_text('#include <stdio.h>\n#include "mosquito_mfcc_int8.h"\nint main(void) {\n'
                    'float input[MOSQUITO_FEATURES];\nwhile (1) {\n'
                    'for (int j=0;j<MOSQUITO_FEATURES;++j) if (scanf("%f",&input[j])!=1) return 0;\n'
                    'printf("%d\\n",mosquito_predict_mfcc(input));\n}\n}\n')
    return header, demo

# %% INT8 evaluation
quantized_predictions = np.full(len(PRIMARY), -1)
quantized_payloads = []
for fold in range(CONFIG["folds"]):
    train = PRIMARY[fold_ids[PRIMARY] != fold]
    locations = np.flatnonzero(fold_ids[PRIMARY] == fold)
    test = PRIMARY[locations]
    payload = quantize_logistic(CLASSICAL_MODELS["MFCC + regressão logística"][fold], X[train][:, MFCC_COLUMNS])
    quantized_payloads.append(payload)
    quantized_predictions[locations] = quantized_predict(payload, X[test][:, MFCC_COLUMNS])
    np.savez_compressed(RESULTS / "models" / f"mfcc_int8_fold{fold}.npz", **payload)
float_predictions = OOF["MFCC + regressão logística"].argmax(1)
quantized_metrics = metrics(Y_SPECIES[PRIMARY], quantized_predictions)
float_metrics = metrics(Y_SPECIES[PRIMARY], float_predictions)
quantization = pd.DataFrame([dict(precision="Logística original (float)", **float_metrics),
                             dict(precision="INT8 / acumulador INT32", **quantized_metrics)])
quantization.to_csv(RESULTS / "int8_metrics.csv", index=False)
display(quantization.round(4))
print("Concordância original/INT8:", round(np.mean(float_predictions == quantized_predictions), 4))
payload = quantized_payloads[0]
payload_bytes = sum(value.nbytes for value in payload.values())
header, c_demo = export_c(payload)
binary = CACHE / "mosquito_mfcc_int8_demo"
compile_check = subprocess.run(["cc", "-O2", str(c_demo), "-lm", "-o", str(binary)], capture_output=True, text=True)
assert compile_check.returncode == 0, compile_check.stderr
c_test = PRIMARY[fold_ids[PRIMARY] == 0][:400]
features = X[c_test][:, MFCC_COLUMNS]
input_text = "\n".join(" ".join(format(float(value), ".9g") for value in row) for row in features) + "\n"
process = subprocess.run([str(binary)], input=input_text, text=True, capture_output=True, check=True)
c_predictions = np.array([int(value) for value in process.stdout.split()])
np.testing.assert_array_equal(c_predictions, quantized_predict(payload, features))
quantization_audit = dict(payload_bytes=payload_bytes, c_python_matching_predictions=len(c_test),
                           prediction_agreement=float(np.mean(float_predictions == quantized_predictions)),
                           original_float_metrics=float_metrics, int8_metrics=quantized_metrics,
                           maximum_int32_accumulator=len(MFCC_COLUMNS) * 127 * 127)
(RESULTS / "int8_audit.json").write_text(json.dumps(quantization_audit, indent=2))
print("Payload de pesos + escalas + padronização:", payload_bytes, "bytes")
print("Exportação C compilada: 400/400 previsões iguais às do Python.")
print("Esses números excluem o frontend de áudio/MFCC, firmware e RAM de processamento.")

# %% Inference function
def classify_recording(path, minimum_score=.7, fold=0):
    path = Path(path).expanduser().resolve()
    wave = read_wave({"path": str(path), "sha256": sha256_file(path)})
    count = len(wave) // SR
    if count < 1:
        raise ValueError("A gravação precisa ter pelo menos 1 segundo.")
    features = np.stack([wave_features(wave[position * SR:(position + 1) * SR])[0]
                         for position in range(count)])
    model = CLASSICAL_MODELS["MFCC + regressão logística"][fold]
    scores = model.predict_proba(features[:, MFCC_COLUMNS]).mean(axis=0)
    ranking = pd.Series(scores, index=SPECIES).sort_values(ascending=False)
    return dict(prediction=str(ranking.index[0]) if ranking.iloc[0] >= minimum_score else "abster",
                score=float(ranking.iloc[0]), seconds_analyzed=count, ranking=ranking.head(5).to_dict())

demo_index = PRIMARY[fold_ids[PRIMARY] == 0][0]
demo_path = ROOT / segments.iloc[demo_index].path
demo_result = classify_recording(demo_path, fold=0)
print("Demonstração com arquivo da dobra 0, ausente do treino do modelo 0:")
print("Rótulo do arquivo:", segments.iloc[demo_index].label)
display(demo_result)

# %% Results and provenance
best = benchmark.sort_values("recording_macro_recall", ascending=False).iloc[0]
print(f"Maior macro recall observado por gravação: {best.model}, {best.recording_macro_recall:.1%}.")
print("A classificação é exploratória: gravação/colônia/aparelho/ambiente não são fatores independentes.")
session_counts = segments.iloc[PRIMARY].groupby("label").session.nunique()
display(session_counts.rename("Sessões inferidas por espécie"))
print("Casos em que uma única sessão inferida concentra a espécie:", session_counts[session_counts == 1].index.tolist())
integrity = {
    "generated_utc": datetime.now(timezone.utc).isoformat(), "config": CONFIG,
    "versions": versions, "python": sys.version, "device": str(DEVICE),
    "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
    "download_total_bytes": sum(r["size"] for r in archives), "verified_archives": len(archives),
    "inventory_files": len(inventory), "usable_candidate_files": len(usable),
    "analyzed_windows": len(segments), "species_windows": len(PRIMARY),
    "species_groups": int(segments.iloc[PRIMARY].group.nunique()), "species": SPECIES.tolist(),
    "group_disjointness_asserted": True, "identical_waveform_disjointness_asserted": True,
    "frontend_maximum_absolute_error": frontend_error, "c_int8_checks": len(c_test),
    "gpu_bitwise_determinism_guaranteed": False,
    "exclusions": inventory.exclusion.replace("", "included").value_counts().to_dict(),
    "pdf_sha256": {f"paper_{k}.pdf": sha256_file(ROOT / "literature" / f"paper_{k}.pdf") for k in [1, 2, 3]
                   if (ROOT / "literature" / f"paper_{k}.pdf").exists()},
    "feature_cache": json.loads((CACHE / "features_manifest.json").read_text()),
}
(RESULTS / "run_manifest.json").write_text(json.dumps(integrity, indent=2, ensure_ascii=False))
files = [p for p in RESULTS.rglob("*") if p.is_file() and p.name != "sha256.json"]
(RESULTS / "sha256.json").write_text(json.dumps({str(p.relative_to(ROOT)): sha256_file(p) for p in files}, indent=2))
print("Resultados, modelos, gráficos e manifests salvos em: results/")
