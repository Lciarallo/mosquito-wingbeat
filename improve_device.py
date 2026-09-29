"""Controlled improvements, weak-label presence detection and calibrated abstention.

Outer folds remain identical to the first experiment. Detector calibration and
decision thresholds use only disjoint groups within each outer training fold.
Presence labels are recording labels, not independently annotated insect events.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import dct
from scipy.ndimage import uniform_filter1d
from scipy.optimize import minimize_scalar
from scipy.special import expit, softmax
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/improvements"
CACHE = ROOT / "data/cache"
SEED, SR, FEATURE_VERSION = 42, 16000, "contrast-delta-1"
METHODS = ["SVM RBF", "ExtraTrees + contraste/dinâmica", "HistGradientBoosting + contraste/dinâmica", "MLP compacta + contraste/dinâmica"]

def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()

def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)

def load_legacy_frontend(root=ROOT):
    config = json.loads((root / "results/run_manifest.json").read_text())["config"]
    code = (root / "notebook_code.py").read_text().split("# %% Feature functions\n", 1)[1].split("# %% Features\n", 1)[0]
    scope = dict(CONFIG=config, SR=config["sample_rate"], np=np, signal=signal, dct=dct)
    exec(compile(code, "legacy_frontend", "exec"), scope)
    return scope["wave_features"], scope["FEATURE_NAMES"]

EXTRA_GRID = np.arange(200, 3901, 50)
EXTRA_NAMES = ([f"contrast_mean_{int(f)}" for f in EXTRA_GRID] +
               [f"contrast_std_{int(f)}" for f in EXTRA_GRID] +
               [f"mfcc_delta_abs_{k}" for k in range(13)] +
               [f"mfcc_delta_std_{k}" for k in range(13)] +
               ["tonal_contrast_mean", "tonal_contrast_std", "peak_track_std", "peak_track_q10", "peak_track_q90"])

def contrast_features(waves, mel):
    waves = np.atleast_2d(np.asarray(waves, dtype="float32"))
    mel = np.asarray(mel, dtype="float32")
    if mel.ndim == 2: mel = mel[None]
    frequencies, _, complex_spectrum = signal.stft(waves, fs=SR, nperseg=2048,
                                                  noverlap=1792, boundary=None, padded=False, axis=-1)
    log_power = 10 * np.log10(np.maximum(np.abs(complex_spectrum)**2, 1e-12))
    # Local spectral contrast suppresses broad microphone coloration. It is not
    # a claim that all noise, or microphone confounding, has been removed.
    contrast = log_power - uniform_filter1d(log_power, size=31, axis=1, mode="nearest")
    means, stds = contrast.mean(axis=-1), contrast.std(axis=-1)
    interpolated_mean = np.stack([np.interp(EXTRA_GRID, frequencies, row) for row in means])
    interpolated_std = np.stack([np.interp(EXTRA_GRID, frequencies, row) for row in stds])
    cepstra = dct(mel*80, type=2, axis=1, norm="ortho")[:, :13]
    delta = np.diff(cepstra, axis=-1)
    band = (frequencies >= 200) & (frequencies <= 900)
    peaks = contrast[:, band, :].max(axis=1)
    tracks = frequencies[band][contrast[:, band, :].argmax(axis=1)]
    summary = np.c_[peaks.mean(axis=1), peaks.std(axis=1), tracks.std(axis=1),
                    np.quantile(tracks, .1, axis=1), np.quantile(tracks, .9, axis=1)]
    return np.c_[interpolated_mean, interpolated_std, np.abs(delta).mean(axis=-1),
                 delta.std(axis=-1), summary].astype("float32")

def improved_wave_features(wave, legacy_frontend=None):
    if legacy_frontend is None: legacy_frontend, _ = load_legacy_frontend()
    base, _, mel, _ = legacy_frontend(wave)
    centered = np.asarray(wave, dtype="float32") - np.mean(wave)
    centered /= max(float(np.sqrt(np.mean(centered**2))), 1e-8)
    return np.r_[base, contrast_features(centered, mel)[0]].astype("float32")

def prepare_data():
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "models").mkdir(exist_ok=True)
    if not (CACHE / "waves.npy").exists():
        raise FileNotFoundError("Execute o notebook principal para baixar e preparar o acervo completo.")
    segments = pd.read_csv(ROOT / "results/segments.csv")
    with np.load(CACHE / "features.npz") as cached: base = cached["features"].copy()
    waves = np.load(CACHE / "waves.npy", mmap_mode="r")
    mel = np.load(CACHE / "mel.npy", mmap_mode="r")
    fingerprint = signature(dict(source=json.loads((CACHE / "features_manifest.json").read_text())["fingerprint"],
                                 version=FEATURE_VERSION, features=EXTRA_NAMES))
    marker, data = CACHE / "improved_features.json", CACHE / "improved_features.npy"
    if marker.exists() and data.exists() and json.loads(marker.read_text())["signature"] == fingerprint:
        extra = np.load(data)
    else:
        blocks = []
        for start in range(0, len(segments), 128):
            blocks.append(contrast_features(waves[start:start+128], mel[start:start+128]))
            if start % 2048 == 0: print(f"Características novas: {start}/{len(segments)}", flush=True)
        extra = np.concatenate(blocks)
        np.save(data, extra)
        atomic_json(marker, dict(signature=fingerprint, version=FEATURE_VERSION, columns=EXTRA_NAMES))
    _, base_names = load_legacy_frontend()
    extended = np.c_[base, extra].astype("float32")
    assert extended.shape == (len(segments), len(base_names)+len(EXTRA_NAMES))
    assert np.isfinite(extended).all()
    for fold_column in ["fold_species", "fold_tinyml"]:
        assert segments.groupby("group")[fold_column].nunique().max() == 1
        assert segments.groupby("content_sha256")[fold_column].nunique().max() == 1
    return segments, base, extended, fingerprint, base_names + EXTRA_NAMES

def weights(indices, labels, segments, class_balance=True):
    frame = pd.DataFrame(dict(group=segments.iloc[indices].group.to_numpy(), label=labels[indices]))
    result = 1 / frame.group.map(frame.groupby("group").size()).to_numpy()
    if class_balance:
        result /= frame.label.map(frame.drop_duplicates("group").groupby("label").size()).to_numpy()
    return result / result.mean()

def split_calibration(indices, segments, seed):
    rng = np.random.default_rng(seed)
    table = segments.iloc[indices][["group", "label"]].drop_duplicates("group")
    reserved = []
    for _, subset in table.groupby("label"):
        names = subset.group.to_numpy().copy(); rng.shuffle(names)
        if len(names) < 2: raise ValueError("Faltam grupos para treino/calibração separados.")
        reserved.extend(names[:min(len(names)-1, max(1, math.ceil(len(names)*.2)))])
    mask = segments.iloc[indices].group.isin(reserved).to_numpy()
    train, calibration = indices[~mask], indices[mask]
    assert set(segments.iloc[train].group).isdisjoint(segments.iloc[calibration].group)
    return train, calibration

def check_disjoint(train, test, segments):
    assert set(segments.iloc[train].group).isdisjoint(segments.iloc[test].group)
    assert set(segments.iloc[train].content_sha256).isdisjoint(segments.iloc[test].content_sha256)

def model_for(name):
    if name == METHODS[0]:
        return make_pipeline(StandardScaler(), SVC(C=3, kernel="rbf", gamma="scale", cache_size=768, random_state=SEED))
    if name == METHODS[1]:
        return ExtraTreesClassifier(n_estimators=200, max_features=.5, min_samples_leaf=2, n_jobs=4, random_state=SEED)
    if name == METHODS[2]:
        # Automatic random-window early stopping is disabled to preserve groups.
        return HistGradientBoostingClassifier(max_iter=120, learning_rate=.08, max_leaf_nodes=15,
                                               min_samples_leaf=20, l2_regularization=1, max_bins=127,
                                               early_stopping=False, random_state=SEED)
    raise ValueError(name)

def fit_classical(model, x, labels, sample_weights):
    if hasattr(model, "steps"):
        model.fit(x, labels, **{model.steps[-1][0]+"__sample_weight":sample_weights})
    else: model.fit(x, labels, sample_weight=sample_weights)
    return model

def predict_proba(model, x):
    if hasattr(model, "predict_proba"): return model.predict_proba(x).astype("float32")
    return softmax(model.decision_function(x), axis=1).astype("float32")

class FeatureMLP(nn.Module):
    def __init__(self, features, classes):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(features, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(.15),
                                     nn.Linear(128, 64), nn.GELU(), nn.Dropout(.1), nn.Linear(64, classes))
    def forward(self, x): return self.network(x)

def mlp_probabilities(model, x, scaler, device):
    model.eval(); parts=[]
    with torch.no_grad():
        for start in range(0, len(x), 512):
            values=torch.from_numpy(scaler.transform(x[start:start+512]).astype("float32")).to(device)
            parts.append(model(values).softmax(-1).cpu().numpy())
    return np.concatenate(parts)

def fit_mlp(x, labels, outer_train, outer_test, segments, classes, path, identity):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4); torch.manual_seed(SEED)
    model = FeatureMLP(x.shape[1], len(classes)).to(device)
    train, validation = split_calibration(outer_train, segments, SEED+len(outer_train))
    check_disjoint(train, validation, segments); check_disjoint(validation, outer_test, segments)
    scaler = StandardScaler().fit(x[train])
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["signature"] == identity:
            model.load_state_dict(saved["state_dict"])
            return mlp_probabilities(model, x[outer_test], scaler, device), saved["history"], saved["training_seconds"]
    sample_weights=weights(train,labels,segments)
    sampler=torch.utils.data.WeightedRandomSampler(torch.from_numpy(sample_weights),len(train),replacement=True,
                                                   generator=torch.Generator().manual_seed(SEED))
    dataset=torch.utils.data.TensorDataset(torch.from_numpy(scaler.transform(x[train]).astype("float32")),
                                          torch.from_numpy(labels[train]).long())
    loader=torch.utils.data.DataLoader(dataset,batch_size=256,sampler=sampler,num_workers=0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    history=[]; best_loss=np.inf; patience=0; best_state=None; begin=time.perf_counter()
    for epoch in range(70):
        model.train()
        for values,target in loader:
            values,target=values.to(device),target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(model(values),target,label_smoothing=.03)
            loss.backward(); optimizer.step()
        probability=mlp_probabilities(model,x[validation],scaler,device)
        loss=float(np.average(-np.log(np.maximum(probability[np.arange(len(validation)),labels[validation]],1e-9)),
                              weights=weights(validation,labels,segments)))
        history.append(dict(epoch=epoch+1,validation_loss=loss))
        if loss < best_loss-1e-4:
            best_loss=loss; patience=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: patience+=1
        if epoch%10==0: print(f"MLP época {epoch+1}: loss={loss:.3f}",flush=True)
        if patience>=10: break
    model.load_state_dict(best_state)
    elapsed=time.perf_counter()-begin
    torch.save(dict(signature=identity,state_dict=best_state,history=history,training_seconds=elapsed,
                    features=x.shape[1],classes=classes.tolist(),scaler_mean=scaler.mean_.tolist(),
                    scaler_scale=scaler.scale_.tolist(),train_groups=segments.iloc[train].group.unique().tolist(),
                    validation_groups=segments.iloc[validation].group.unique().tolist(),
                    test_groups=segments.iloc[outer_test].group.unique().tolist()),path)
    return mlp_probabilities(model,x[outer_test],scaler,device),history,elapsed

def group_predictions(indices, targets, probability, classes, segments):
    frame=pd.DataFrame(probability,index=segments.iloc[indices].group.to_numpy())
    scores=frame.groupby(level=0,sort=True).mean()
    true=pd.Series(targets,index=segments.iloc[indices].group.to_numpy()).groupby(level=0,sort=True).first()
    result=pd.DataFrame(dict(group=scores.index,true=true.to_numpy(),predicted=scores.to_numpy().argmax(1)))
    result["true_label"]=classes[result.true]; result["predicted_label"]=classes[result.predicted]
    return result

def summarize(name, targets, probability, recordings, training_seconds):
    prediction=probability.argmax(1)
    return dict(model=name,accuracy=accuracy_score(targets,prediction),
                balanced_accuracy=balanced_accuracy_score(targets,prediction),
                macro_f1=f1_score(targets,prediction,average="macro",zero_division=0),
                recording_macro_recall=balanced_accuracy_score(recordings.true,recordings.predicted),
                n_windows=len(targets),n_groups=len(recordings),training_seconds=training_seconds)

def benchmark_species(segments,base,extended,fingerprint,names):
    mask=(segments.fold_species>=0)&segments.label.ne("noise")
    indices=np.flatnonzero(mask); classes=np.sort(segments.loc[mask,"label"].unique())
    labels=np.full(len(segments),-1); labels[mask]=np.searchsorted(classes,segments.loc[mask,"label"])
    results=[]; oof={}; histories={}
    original=pd.read_csv(ROOT/"results/species_metrics.csv")
    baseline=original.loc[original.model.eq("MFCC/PSD/F0 + ExtraTrees")].iloc[0].to_dict()
    baseline["model"]="ExtraTrees original (referência)"; results.append(baseline)
    for method_id,name in enumerate(METHODS):
        x=base if name==METHODS[0] else extended
        probability=np.zeros((len(indices),len(classes)),dtype="float32"); elapsed=0
        for fold in range(3):
            train=indices[segments.iloc[indices].fold_species.to_numpy()!=fold]
            positions=np.flatnonzero(segments.iloc[indices].fold_species.to_numpy()==fold); test=indices[positions]
            check_disjoint(train,test,segments)
            spec=model_for(name).get_params() if name!=METHODS[3] else dict(epochs=70,patience=10,architecture="128-64",lr=.001)
            # String form covers sklearn objects in Pipeline parameters.
            identity=signature(dict(features=fingerprint,method=name,params=str(spec),train=train.tolist(),test=test.tolist()))
            path=OUT/"models"/f"species_{method_id}_fold{fold}.joblib"
            print(f"{name}, dobra {fold}: {len(train)} treino / {len(test)} teste",flush=True)
            if name==METHODS[3]:
                prediction,history,seconds=fit_mlp(x,labels,train,test,segments,classes,path.with_suffix(".pt"),identity)
                histories[f"fold{fold}"]=history; elapsed+=seconds
            elif path.exists() and (saved:=joblib.load(path))["signature"]==identity:
                prediction=saved["test_probabilities"]; elapsed+=saved["training_seconds"]
            else:
                begin=time.perf_counter(); model=fit_classical(model_for(name),x[train],labels[train],weights(train,labels,segments))
                prediction=predict_proba(model,x[test]); seconds=time.perf_counter()-begin; elapsed+=seconds
                joblib.dump(dict(model=model,signature=identity,test_probabilities=prediction,training_seconds=seconds),path,compress=3)
            probability[positions]=prediction
        oof[name]=probability
        recordings=group_predictions(indices,labels[indices],probability,classes,segments)
        recordings.to_csv(OUT/f"species_groups_{method_id}.csv",index=False)
        row=summarize(name,labels[indices],probability,recordings,elapsed); results.append(row)
        pd.DataFrame(results).to_csv(OUT/"species_metrics.csv",index=False)
        print("Resultado:",json.dumps(row,ensure_ascii=False),flush=True)
    # Equal weights are fixed before looking at outer-fold predictions.
    ensemble=np.mean([oof[name] for name in METHODS[:3]],axis=0)
    recordings=group_predictions(indices,labels[indices],ensemble,classes,segments)
    recordings.to_csv(OUT/"species_groups_ensemble.csv",index=False)
    results.append(summarize("Ensemble fixo (RBF + árvores + boosting)",labels[indices],ensemble,recordings,0))
    oof["ensemble"]=ensemble
    pd.DataFrame(results).to_csv(OUT/"species_metrics.csv",index=False)
    np.savez_compressed(OUT/"species_oof.npz",**{f"model_{i}":value for i,value in enumerate(oof.values())})
    atomic_json(OUT/"species_protocol.json",dict(method_order=list(oof),classes=classes.tolist(),config_feature_version=FEATURE_VERSION,
                feature_names=names,feature_signature=fingerprint,outer_fold_column="fold_species",seed=SEED,
                mlp_history=histories,group_disjointness=True,identical_waveform_disjointness=True,
                svm_scores_calibrated=False,ensemble_weights=[1/3,1/3,1/3],adaptive_analysis_of_previously_used_corpus=True))
    return pd.DataFrame(results)

def temperature_scale(probability, temperature):
    return softmax(np.log(np.maximum(probability,1e-9))/temperature,axis=1)

def detector_statistics(targets, detected, groups, score=None):
    def mean(mask,values):
        frame=pd.DataFrame(dict(group=np.asarray(groups)[mask],value=np.asarray(values)[mask]))
        return float(frame.groupby("group").value.mean().mean())
    positives=targets==1; negatives=~positives
    result=dict(window_recall=float(np.mean(detected[positives])),window_false_positive_rate=float(np.mean(detected[negatives])),
                source_mean_recall=mean(positives,detected),source_mean_false_positive_rate=mean(negatives,detected),
                window_balanced_accuracy=balanced_accuracy_score(targets,detected),
                positive_windows=int(sum(positives)),noise_windows=int(sum(negatives)),
                positive_groups=len(set(np.asarray(groups)[positives])),noise_groups=len(set(np.asarray(groups)[negatives])))
    if score is not None: result["roc_auc"]=roc_auc_score(targets,score)
    return result

def operating_threshold(scores,negative_indices,segments,target_fpr=.05):
    w=weights(negative_indices,np.zeros(len(segments)),segments,class_balance=False)
    candidates=np.r_[0,np.unique(scores),np.nextafter(1.,2.)]
    for threshold in candidates:
        if np.average(scores>=threshold,weights=w)<=target_fpr:
            return float(threshold)
    return float(np.nextafter(1.,2.))

def presence_and_abstention(segments,extended,fingerprint,names):
    indices=np.flatnonzero(segments.fold_species.to_numpy()>=0)
    labels=(segments.label.ne("noise")).to_numpy().astype(int)
    classes=np.sort(segments.loc[segments.label.ne("noise"),"label"].unique())
    species_labels=np.full(len(segments),-1)
    positive=segments.label.ne("noise").to_numpy(); species_labels[positive]=np.searchsorted(classes,segments.loc[positive,"label"])
    fold_rows=[]; output=[]; selection_rows=[]
    for fold in range(3):
        outer_train=indices[segments.iloc[indices].fold_species.to_numpy()!=fold]
        test=indices[segments.iloc[indices].fold_species.to_numpy()==fold]
        fit,cal=split_calibration(outer_train,segments,SEED+fold)
        check_disjoint(fit,cal,segments); check_disjoint(fit,test,segments); check_disjoint(cal,test,segments)
        identity=signature(dict(features=fingerprint,protocol="presence-hgb-cal-1",fit=fit.tolist(),cal=cal.tolist(),test=test.tolist()))
        path=OUT/"models"/f"device_fold{fold}.joblib"
        if path.exists() and (cached:=joblib.load(path))["signature"]==identity:
            bundle=cached["bundle"]
        else:
            print(f"Detecção/calibração, dobra {fold}",flush=True)
            detector=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=3000,random_state=SEED))
            fit_classical(detector,extended[fit],labels[fit],weights(fit,labels,segments))
            calibration=LogisticRegression(C=100,max_iter=1000,random_state=SEED)
            raw=detector.decision_function(extended[cal])[:,None]
            calibration.fit(raw,labels[cal],sample_weight=weights(cal,labels,segments,class_balance=False))
            presence_score=calibration.predict_proba(raw)[:,1]
            negative=cal[labels[cal]==0]
            threshold=operating_threshold(presence_score[labels[cal]==0],negative,segments)
            species_fit=fit[labels[fit]==1]; species_cal=cal[labels[cal]==1]
            classifier=fit_classical(model_for(METHODS[2]),extended[species_fit],species_labels[species_fit],
                                     weights(species_fit,species_labels,segments))
            probability=predict_proba(classifier,extended[species_cal]); w=weights(species_cal,species_labels,segments,class_balance=False)
            def calibration_loss(temp):
                p=temperature_scale(probability,temp)
                return np.average(-np.log(np.maximum(p[np.arange(len(species_cal)),species_labels[species_cal]],1e-9)),weights=w)
            temperature=float(minimize_scalar(calibration_loss,bounds=(.25,10),method="bounded").x)
            p=temperature_scale(probability,temperature); correct=p.argmax(1)==species_labels[species_cal]
            # Threshold target is selected on calibration only; no guarantee on unseen recordings.
            species_threshold=float(np.nextafter(1.,2.))
            for candidate in np.linspace(.05,.99,95):
                accepted=p.max(1)>=candidate
                groups=segments.iloc[species_cal[accepted]].group.nunique()
                if groups>=15 and np.average(accepted,weights=w)>=.1 and np.average(correct[accepted],weights=w[accepted])>=.8:
                    species_threshold=float(candidate); break
            bundle=dict(detector=detector,calibration=calibration,classifier=classifier,
                        presence_threshold=threshold,species_threshold=species_threshold,temperature=temperature,
                        classes=classes.tolist(),feature_names=names,feature_version=FEATURE_VERSION,
                        sample_rate=SR,window_samples=SR,
                        threshold_targets=dict(calibration_source_false_positive_rate=.05,calibration_accepted_accuracy=.8),
                        fit_groups=segments.iloc[fit].group.unique().tolist(),calibration_groups=segments.iloc[cal].group.unique().tolist(),
                        test_groups=segments.iloc[test].group.unique().tolist(),labels_are_weak=True,fold=fold)
            joblib.dump(dict(signature=identity,bundle=bundle),path,compress=3)
        score=bundle["calibration"].predict_proba(bundle["detector"].decision_function(extended[test])[:,None])[:,1]
        detected=score>=bundle["presence_threshold"]
        probability=temperature_scale(predict_proba(bundle["classifier"],extended[test]),bundle["temperature"])
        prediction=probability.argmax(1); confidence=probability.max(1)
        accepted=detected&(confidence>=bundle["species_threshold"])
        for mode,decision in [("Limiar fixo 0,5",score>=.5),("Controle de alarmes por calibração",detected)]:
            row=detector_statistics(labels[test],decision,segments.iloc[test].group.to_numpy(),score)
            fold_rows.append(dict(fold=fold,mode=mode,presence_threshold=bundle["presence_threshold"],**row))
        selection_rows.append(dict(fold=fold,species_threshold=bundle["species_threshold"],temperature=bundle["temperature"],
                                  accepted_windows=int(sum(accepted)),coverage_positive_windows=float(np.mean(accepted[labels[test]==1])),
                                  accepted_species_accuracy=float(np.mean(prediction[accepted]==species_labels[test][accepted])) if accepted.any() else None))
        frame=segments.iloc[test][["path","group","label","start_s"]].copy()
        frame["fold"]=fold; frame["presence_score"]=score; frame["detected"]=detected
        frame["species_predicted"]=classes[prediction]; frame["species_score"]=confidence; frame["species_accepted"]=accepted
        output.append(frame)
        # Publish compact detector coefficients; full larger classifiers remain locally cached.
        scaler,linear=bundle["detector"].steps[0][1],bundle["detector"].steps[-1][1]
        np.savez_compressed(OUT/"models"/f"presence_linear_fold{fold}.npz",mean=scaler.mean_,scale=scaler.scale_,
                            weights=linear.coef_,bias=linear.intercept_,calibration_weights=bundle["calibration"].coef_,
                            calibration_bias=bundle["calibration"].intercept_,threshold=bundle["presence_threshold"])
        print("Detecção:",json.dumps(fold_rows[-1],ensure_ascii=False),flush=True)
    pd.DataFrame(fold_rows).to_csv(OUT/"presence_metrics.csv",index=False)
    pd.DataFrame(selection_rows).to_csv(OUT/"abstention_metrics.csv",index=False)
    combined=pd.concat(output).sort_index(); combined.to_csv(OUT/"device_predictions.csv",index=False)
    aggregate=[]
    for mode in ["Limiar fixo 0,5","Controle de alarmes por calibração"]:
        decisions=combined.presence_score.to_numpy()>=.5 if mode=="Limiar fixo 0,5" else combined.detected.to_numpy()
        aggregate.append(dict(mode=mode,**detector_statistics((combined.label!="noise").to_numpy().astype(int),
                                                              decisions,combined.group.to_numpy(),combined.presence_score.to_numpy())))
    pd.DataFrame(aggregate).to_csv(OUT/"presence_pooled_metrics.csv",index=False)
    accepted=combined.loc[combined.species_accepted]
    atomic_json(OUT/"device_protocol.json",dict(features=fingerprint,feature_version=FEATURE_VERSION,classes=classes.tolist(),
                selected_classifier=METHODS[2],labels_are_weak=True,independent_event_annotations=False,
                train_calibration_test_group_disjointness=True,thresholds_selected_without_outer_test=True,
                calibrated_presence_prior="empirical recording-group mixture, not field prevalence",
                detector_source_false_positive_target=.05,accepted_species_validation_accuracy_target=.8,
                pooled_species_accepted_accuracy=float(np.mean(accepted.label==accepted.species_predicted)) if len(accepted) else None,
                pooled_species_accepted_windows=len(accepted),
                pooled_positive_coverage=float(combined.loc[combined.label!="noise","species_accepted"].mean()),
                inference_default_fold=0,hardware_runtime_measured=False))

def paired_comparisons():
    baseline=pd.read_csv(ROOT/"results/predictions_classical_6.csv").set_index("group")
    rows=[]
    for method_id,name in list(enumerate(METHODS))+[("ensemble","Ensemble fixo (RBF + árvores + boosting)")]:
        current=pd.read_csv(OUT/f"species_groups_{method_id}.csv").set_index("group").loc[baseline.index]
        assert np.array_equal(current.true_label,baseline.true_label)
        pieces=[]
        for label,subset in baseline.groupby("true_label"):
            ids=subset.index
            pieces.append(((current.loc[ids].true_label==current.loc[ids].predicted_label).to_numpy().astype(float)-
                           (baseline.loc[ids].true_label==baseline.loc[ids].predicted_label).to_numpy().astype(float)))
        rng=np.random.default_rng(SEED); draws=[]
        for _ in range(2000): draws.append(np.mean([p[rng.integers(0,len(p),len(p))].mean() for p in pieces]))
        lower,upper=np.quantile(draws,[.025,.975])
        rows.append(dict(model=name,paired_macro_recall_gain=float(np.mean([p.mean() for p in pieces])),
                         descriptive_ci_low=lower,descriptive_ci_high=upper))
    pd.DataFrame(rows).to_csv(OUT/"paired_group_differences.csv",index=False)

def main(phase="all"):
    begin=time.perf_counter(); segments,base,extended,fingerprint,names=prepare_data()
    if phase in ["all","species"]: benchmark_species(segments,base,extended,fingerprint,names); paired_comparisons()
    if phase in ["all","device"]: presence_and_abstention(segments,extended,fingerprint,names)
    atomic_json(OUT/"execution.json",dict(phase=phase,elapsed_seconds=time.perf_counter()-begin,feature_signature=fingerprint,
                script_sha256=file_hash(ROOT/"improve_device.py"),python_versions=dict(numpy=np.__version__,torch=torch.__version__),
                same_outer_folds_as_original=True,new_features=len(EXTRA_NAMES),total_features=len(names),
                physical_device_tested=False,field_false_alarms_per_hour_measured=False))
    print("Melhorias executadas. Resultados:",OUT,flush=True)

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=["all","species","device"],default="all")
    main(parser.parse_args().phase)
