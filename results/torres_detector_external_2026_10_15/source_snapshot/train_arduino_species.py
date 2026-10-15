"""Fit, calibrate, evaluate and export 20-species models for Nano 33 BLE Sense.

Requires the complete dataset/cache prepared by the notebook. No test groups
are used for model choice, epoch selection, temperature or rejection thresholds.
The default fold is chosen by calibration eligibility, never by test accuracy.
It is not a production retrain on all evaluated data.
"""
from __future__ import annotations
import json
import math
import shutil
import time
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
from improve_device import ROOT,SEED,weights,check_disjoint,atomic_json,file_hash,signature
from train_arduino import prepare,calibrated
from arduino_species_models import TinySpeciesMLP,fused_layers,portable_logits,probabilities,confirm_species

OUT = ROOT/"results/arduino_species"
FIRMWARE = ROOT/"firmware/MosquitoSpecies"
NAMES = ["Logística 68 → 20","MLP 68 → 64 → 20","MLP 68 → 128 → 64 → 20"]
VERSION = "source-disjoint-species-streaming-1"


def model_for(candidate):
    if candidate==0:
        return make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=SEED))
    return TinySpeciesMLP((64,) if candidate==1 else (128,64),seed=SEED)


def inner_split(indices,segments,seed):
    """Singleton classes stay in training; independent validation is impossible."""
    rng = np.random.default_rng(seed)
    table = segments.iloc[indices][["group","label"]].drop_duplicates("group")
    reserved,unvalidated = [],[]
    for label,subset in table.groupby("label"):
        groups = subset.group.to_numpy().copy()
        rng.shuffle(groups)
        if len(groups)<2:
            unvalidated.append(label)
        else:
            reserved.extend(groups[:min(len(groups)-1,max(1,math.ceil(len(groups)*.2)))])
    mask = segments.iloc[indices].group.isin(reserved).to_numpy()
    return indices[~mask],indices[mask],unvalidated


def fit(model,x,y,w,validation=None,fixed_epochs=None):
    if isinstance(model,TinySpeciesMLP):
        return model.fit(x,y,w,validation=validation,fixed_epochs=fixed_epochs)
    return model.fit(x,y,logisticregression__sample_weight=w)


def calibrate(bundle,x,y,w,groups):
    logits = portable_logits(bundle,x).astype("float64")
    def loss(log_temperature):
        z = logits/np.exp(log_temperature)
        return np.average(logsumexp(z,axis=1)-z[np.arange(len(y)),y],weights=w)
    result = minimize_scalar(loss,bounds=(np.log(.25),np.log(4.)),method="bounded")
    assert result.success
    bundle["temperature"] = float(np.float32(np.exp(result.x)))
    confidence = probabilities(bundle,x).max(axis=1)
    correct = logits.argmax(axis=1)==y
    # Select the highest coverage meeting a descriptive calibration target.
    # A model can reject everything if the target cannot be met.
    threshold,candidates = 1.0001,[]
    grid = np.unique(np.r_[np.linspace(.05,.99,189),np.quantile(confidence,np.linspace(0,1,101))])
    for value in grid:
        value = float(np.nextafter(np.float32(value),np.float32(np.inf)))
        accepted = confidence>=value
        coverage = float(np.sum(w[accepted])/np.sum(w))
        count_groups = int(len(np.unique(groups[accepted])))
        precision = float(np.average(correct[accepted],weights=w[accepted])) if accepted.any() else None
        candidates.append(dict(threshold=value,source_class_weighted_coverage=coverage,
                               source_class_weighted_accuracy=precision,accepted_groups=count_groups))
        if precision is not None and precision>=.8 and coverage>=.1 and count_groups>=15:
            threshold = value
            break
    bundle["confidence_threshold"] = threshold
    bundle["calibration"] = dict(temperature=bundle["temperature"],threshold=threshold,
        target_source_class_weighted_accuracy=.8,minimum_weighted_coverage=.1,minimum_accepted_groups=15,
        target_met=threshold<=1,threshold_search=candidates,
        note="Calibration target is not a guarantee of correctness on test or Arduino recordings.")


def classification_metrics(frame,p,classes):
    positive = frame.label.ne("noise").to_numpy()
    current = frame.loc[positive].copy()
    p = p[positive]
    y = current.label.map({c:i for i,c in enumerate(classes)}).to_numpy()
    prediction = p.argmax(axis=1)
    source_p = pd.DataFrame(p,index=current.group.to_numpy()).groupby(level=0,sort=False).mean()
    source_y = current.groupby("group",sort=False).label.first().map({c:i for i,c in enumerate(classes)})
    source_y = source_y.reindex(source_p.index).to_numpy()
    return dict(accuracy=float(accuracy_score(y,prediction)),
                balanced_accuracy=float(balanced_accuracy_score(y,prediction)),
                macro_f1=float(f1_score(y,prediction,average="macro",zero_division=0)),
                recording_accuracy=float(accuracy_score(source_y,source_p.to_numpy().argmax(axis=1))),
                recording_macro_recall=float(balanced_accuracy_score(source_y,source_p.to_numpy().argmax(axis=1))),
                positive_windows=len(y),positive_groups=current.group.nunique())


def selective_metrics(frame,accepted,classes):
    target = frame.label.map({c:i for i,c in enumerate(classes)}).fillna(-1).to_numpy().astype(int)
    positive = target>=0
    emitted = accepted>=0
    correct = accepted==target
    def mean_by_source(mask):
        if not mask.any():
            return None
        return float(pd.DataFrame(dict(group=frame.loc[mask,"group"].to_numpy(),
                                      accepted=emitted[mask])).groupby("group").accepted.mean().mean())
    return dict(positive_windows=int(positive.sum()),noise_windows=int((~positive).sum()),
        positive_groups=frame.loc[positive,"group"].nunique(),noise_groups=frame.loc[~positive,"group"].nunique(),
        positive_coverage=float(emitted[positive].mean()) if positive.any() else None,
        noise_false_species_rate=float(emitted[~positive].mean()) if (~positive).any() else None,
        positive_source_mean_coverage=mean_by_source(positive),noise_source_mean_false_species_rate=mean_by_source(~positive),
        accepted_windows=int(emitted.sum()),accepted_positive_windows=int((emitted&positive).sum()),
        accepted_noise_windows=int((emitted&~positive).sum()),
        accepted_accuracy_including_noise=float(correct[emitted].mean()) if emitted.any() else None,
        accepted_positive_class_accuracy=float(correct[emitted&positive].mean()) if (emitted&positive).any() else None,
        correct_positive_fraction=float((emitted&correct)[positive].mean()))


def literal(value):
    value = f"{float(value):.9g}"
    if "." not in value and "e" not in value:
        value += ".0"
    return value+"f"


def export(bundle,presence,fold,classes):
    lines = ["// Generated by train_arduino_species.py. Experimental, source-disjoint fold model.",
             "#ifndef MOSQUITO_SPECIES_MODEL_H","#define MOSQUITO_SPECIES_MODEL_H",
             '#include "StreamingFeatures.h"',"namespace mosquito {",
             "static const uint8_t kSpeciesCount=20;",
             f"static const uint8_t kSpeciesFold={fold};",
             "static const char* const kSpeciesNames[kSpeciesCount]={"+
                 ",".join(json.dumps(c) for c in classes)+"};",
             "static const float kSpeciesTemperature="+literal(bundle["temperature"])+";",
             "static const float kSpeciesConfidenceThreshold="+literal(bundle["confidence_threshold"])+";"]
    payload = dict(classes=np.array(classes),temperature=np.float32(bundle["temperature"]),
                   confidence_threshold=np.float32(bundle["confidence_threshold"]),fold=fold,
                   presence_threshold=np.float32(presence["threshold_10"]))
    for i,(w,b) in enumerate(zip(bundle["weights"],bundle["biases"])):
        lines.append(f"static const float kSpeciesWeights{i}[{w.shape[0]}][{w.shape[1]}]={{")
        lines.extend("  {"+",".join(literal(v) for v in row)+"}," for row in w)
        lines.extend(["};",f"static const float kSpeciesBias{i}[{len(b)}]={{"+",".join(literal(v) for v in b)+"};"])
        payload[f"weights{i}"],payload[f"bias{i}"] = w,b
    lines.append("inline void speciesProbabilities(const float* x,float* out) {")
    previous = "x"
    for i,(w,b) in enumerate(zip(bundle["weights"],bundle["biases"])):
        name = "out" if i==len(bundle["weights"])-1 else f"hidden{i}"
        if name!="out":
            lines.append(f"  float {name}[{w.shape[0]}];")
        lines.extend([f"  for(uint16_t h=0;h<{w.shape[0]};++h) {{ float value=kSpeciesBias{i}[h];",
                      f"    for(uint16_t j=0;j<{w.shape[1]};++j) value+=kSpeciesWeights{i}[h][j]*{previous}[j];",
                      f"    {name}[h]="+("value; }" if name=="out" else "fmaxf(0.f,value); }")])
        previous = name
    lines.extend(["  float maximum=out[0];",
                  "  for(uint8_t c=1;c<kSpeciesCount;++c) maximum=fmaxf(maximum,out[c]);",
                  "  float total=0.f;",
                  "  for(uint8_t c=0;c<kSpeciesCount;++c) { out[c]=expf((out[c]-maximum)/kSpeciesTemperature);total+=out[c]; }",
                  "  for(uint8_t c=0;c<kSpeciesCount;++c) out[c]/=total;","}",
                  "inline int8_t bestSpecies(const float* p) {",
                  "  int8_t best=0;for(uint8_t c=1;c<kSpeciesCount;++c) if(p[c]>p[best]) best=c;return best; }",
                  "} // namespace mosquito","#endif"])
    header = FIRMWARE/f"SpeciesModelFold{fold}.h"
    header.write_text("\n".join(lines)+"\n")
    # Complete portable numeric checkpoint, including the binary gate.
    gate = presence["model"]
    coef = gate.w1/gate.scaler.scale_[None]
    payload.update(presence_weights0=coef.astype("float32"),
                   presence_bias0=(gate.b1-coef@gate.scaler.mean_).astype("float32"),
                   presence_weights1=gate.w2,presence_bias1=np.float32(gate.b2),
                   presence_calibration_slope=np.float32(presence["calibration_slope"]),
                   presence_calibration_bias=np.float32(presence["calibration_bias"]))
    np.savez_compressed(OUT/"models"/f"species_fold{fold}.npz",**payload)
    return dict(fold=fold,header=str(header.relative_to(ROOT)),sha256=file_hash(header),
                model_name=bundle["model_name"],dimensions=[68]+[w.shape[0] for w in bundle["weights"]],
                species_parameter_count=sum(w.size+b.size for w,b in zip(bundle["weights"],bundle["biases"])),
                species_numeric_bytes=4*(sum(w.size+b.size for w,b in zip(bundle["weights"],bundle["biases"]))+2),
                temperature=bundle["temperature"],confidence_threshold=bundle["confidence_threshold"])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"models").mkdir(exist_ok=True)
    FIRMWARE.mkdir(parents=True,exist_ok=True)
    segments,x,identity = prepare()
    parent = json.loads((ROOT/"results/arduino/protocol.json").read_text())
    classes = sorted(segments.loc[segments.label.ne("noise"),"label"].unique().tolist())
    assert len(classes)==20
    y = segments.label.map({c:i for i,c in enumerate(classes)}).fillna(-1).to_numpy().astype(int)
    all_frames = [[] for _ in range(4)]
    all_probabilities = [[] for _ in range(4)]
    split_records,exports,fold_metrics,selected_parts,selected_p = [],[],[],[],[]
    for fold in range(3):
        source = parent["source_splits"][fold]
        assert source["fold"]==fold
        # Exact parent splits ensure the presence gate has never trained on
        # species calibration groups or on outer test groups.
        fit_idx = np.flatnonzero(segments.group.isin(source["fit_groups"]).to_numpy()&(y>=0))
        cal_idx = np.flatnonzero(segments.group.isin(source["calibration_groups"]).to_numpy()&(y>=0))
        test = np.flatnonzero(segments.fold_species.to_numpy()==fold)
        inner,validation,unsupported = inner_split(fit_idx,segments,SEED+200+fold)
        for a,b in [(fit_idx,cal_idx),(fit_idx,test),(cal_idx,test),(inner,validation),(validation,test)]:
            check_disjoint(a,b,segments)
        assert len(np.unique(y[inner]))==20
        candidate_bundles = []
        for candidate in range(3):
            path = OUT/"models"/f"candidate{candidate}_fold{fold}.joblib"
            sig = signature(dict(version=VERSION,features=identity,
                source_sha256=file_hash(ROOT/"arduino_species_models.py"),candidate=candidate,
                fit=fit_idx.tolist(),cal=cal_idx.tolist(),inner=inner.tolist(),validation=validation.tolist(),
                params=str(model_for(candidate).get_params()),test=test.tolist()))
            if path.exists() and (saved:=joblib.load(path))["signature"]==sig:
                bundle = saved["bundle"]
            else:
                print(f"Espécies Arduino: {NAMES[candidate]}, dobra {fold}",flush=True)
                begin = time.perf_counter()
                temporary = fit(model_for(candidate),x[inner],y[inner],weights(inner,y,segments),
                    validation=(x[validation],y[validation],weights(validation,y,segments)))
                logits = temporary.decision_function(x[validation])
                correct = logits.argmax(axis=1)==y[validation]
                inner_accuracy = float(np.average(correct,weights=weights(validation,y,segments)))
                model = fit(model_for(candidate),x[fit_idx],y[fit_idx],weights(fit_idx,y,segments),
                            fixed_epochs=getattr(temporary,"best_epoch",None))
                ww,bb = fused_layers(model)
                bundle = dict(model=model,model_name=NAMES[candidate],weights=ww,biases=bb,
                    inner_source_class_weighted_accuracy=inner_accuracy,classes=classes,
                    neural_epoch=getattr(temporary,"best_epoch",None),
                    neural_history=getattr(temporary,"history",None),fold=fold)
                calibrate(bundle,x[cal_idx],y[cal_idx],weights(cal_idx,y,segments),
                          segments.iloc[cal_idx].group.to_numpy())
                bundle["training_seconds"] = time.perf_counter()-begin
                joblib.dump(dict(signature=sig,bundle=bundle),path,compress=3)
            candidate_bundles.append(bundle)
        winner = int(np.argmax([b["inner_source_class_weighted_accuracy"] for b in candidate_bundles]))
        chosen = candidate_bundles[winner]
        presence = joblib.load(ROOT/"results/arduino/models"/f"candidate{source['selected_candidate']}_fold{fold}.joblib")["bundle"]
        exports.append(export(chosen,presence,fold,classes))
        split_records.append(dict(fold=fold,selected_candidate=winner,selected_name=NAMES[winner],
            inner_accuracy={b["model_name"]:b["inner_source_class_weighted_accuracy"] for b in candidate_bundles},
            selected_epoch=chosen["neural_epoch"],fit_groups=segments.iloc[fit_idx].group.unique().tolist(),
            calibration_groups=segments.iloc[cal_idx].group.unique().tolist(),
            inner_validation_groups=segments.iloc[validation].group.unique().tolist(),
            classes_without_independent_inner_validation=unsupported,test_groups=source["test_groups"],
            calibration=chosen["calibration"],presence_candidate=source["selected_candidate"]))
        for candidate,bundle in enumerate(candidate_bundles):
            p = probabilities(bundle,x[test])
            for index in ([candidate,3] if candidate==winner else [candidate]):
                all_frames[index].append(segments.iloc[test])
                all_probabilities[index].append(p)
            if candidate==winner:
                frame = segments.iloc[test][["path","group","label","start_s"]].copy()
                frame["segment_index"],frame["fold"] = test,fold
                frame["candidate"] = p.argmax(axis=1)
                frame["class_score"] = p.max(axis=1)
                frame["class_threshold"] = bundle["confidence_threshold"]
                frame["presence_score"] = calibrated(presence,x[test])
                frame["presence_threshold"] = presence["threshold_10"]
                frame["eligible_candidate"] = np.where(
                    (frame.class_score>=frame.class_threshold)&(frame.presence_score>=frame.presence_threshold),
                    frame.candidate,-1)
                selected_parts.append(frame)
                selected_p.append(p)
                fold_metrics.append(dict(fold=fold,model=NAMES[winner],**classification_metrics(frame,p,classes)))
    rows = []
    for i,name in enumerate(NAMES+["Escolha por validação interna"]):
        frame,p = pd.concat(all_frames[i]),np.concatenate(all_probabilities[i])
        rows.append(dict(model=name,**classification_metrics(frame,p,classes)))
    pd.DataFrame(rows).to_csv(OUT/"metrics.csv",index=False)
    pd.DataFrame(fold_metrics).to_csv(OUT/"fold_metrics.csv",index=False)
    prediction = pd.concat(selected_parts,ignore_index=True)
    p = np.concatenate(selected_p)
    positive = prediction.label.ne("noise").to_numpy()
    target = prediction.loc[positive,"label"].map({c:i for i,c in enumerate(classes)}).to_numpy()
    matrix = confusion_matrix(target,p[positive].argmax(axis=1),labels=np.arange(20))
    pd.DataFrame(matrix,index=classes,columns=classes).to_csv(OUT/"confusion_matrix.csv",index_label="true_species")
    per_class = []
    for c,name in enumerate(classes):
        true,guessed = int(matrix[c].sum()),int(matrix[:,c].sum())
        per_class.append(dict(species=name,windows=true,groups=prediction.loc[prediction.label.eq(name),"group"].nunique(),
            recall=float(matrix[c,c]/true),precision=float(matrix[c,c]/guessed) if guessed else 0.))
    pd.DataFrame(per_class).to_csv(OUT/"per_class.csv",index=False)
    operating = [dict(rule="Uma janela: presença + confiança",**selective_metrics(
        prediction,prediction.eligible_candidate.to_numpy(),classes))]
    temporal = []
    for _,part in prediction.groupby("fold",sort=False):
        part = part.sort_values(["path","start_s"]).copy()
        labels,eligible = confirm_species(part.eligible_candidate,part.path,part.start_s)
        part["confirmed_candidate"],part["contiguous_endpoint"] = labels,eligible
        temporal.append(part)
    temporal = pd.concat(temporal)
    subset = temporal.loc[temporal.contiguous_endpoint]
    for rule,column in [("Uma janela nos endpoints contíguos","eligible_candidate"),
                        ("2/3 mesma espécie nos endpoints contíguos","confirmed_candidate")]:
        operating.append(dict(rule=rule,**selective_metrics(subset,subset[column].to_numpy(),classes)))
    pd.DataFrame(operating).to_csv(OUT/"selective_metrics.csv",index=False)
    temporal.sort_values("segment_index").to_csv(OUT/"predictions.csv.gz",index=False,
        compression=dict(method="gzip",mtime=0))
    np.savez_compressed(OUT/"oof_probabilities.npz",probabilities=p.astype("float32"),
                        segment_index=prediction.segment_index.to_numpy(),classes=np.array(classes))
    # Choose a deployable prototype using calibration eligibility only.
    # This does not select by the outer test accuracy. Other folds rejected
    # everything when the predeclared calibration target could not be met.
    calibrated_folds = [e["fold"] for e in exports if e["confidence_threshold"]<=1.]
    default_fold = calibrated_folds[0] if calibrated_folds else 0
    shutil.copyfile(FIRMWARE/f"SpeciesModelFold{default_fold}.h",FIRMWARE/"SpeciesModel.h")
    shutil.copyfile(ROOT/"firmware/MosquitoPresence/StreamingFeatures.h",FIRMWARE/"StreamingFeatures.h")
    for fold in range(3):
        original = ROOT/"firmware/MosquitoPresence"/("PresenceModel.h" if fold==0 else f"PresenceModelFold{fold}.h")
        shutil.copyfile(original,FIRMWARE/f"PresenceModelFold{fold}.h")
    shutil.copyfile(FIRMWARE/f"PresenceModelFold{default_fold}.h",FIRMWARE/"PresenceModel.h")
    default = temporal.loc[temporal.fold.eq(default_fold)]
    default_subset = default.loc[default.contiguous_endpoint]
    default_rows = [dict(rule="Uma janela: presença + confiança",**selective_metrics(
        default,default.eligible_candidate.to_numpy(),classes))]
    for rule,column in [("Uma janela nos endpoints contíguos","eligible_candidate"),
                        ("2/3 mesma espécie nos endpoints contíguos","confirmed_candidate")]:
        default_rows.append(dict(rule=rule,**selective_metrics(default_subset,default_subset[column].to_numpy(),classes)))
    pd.DataFrame(default_rows).to_csv(OUT/"default_selective_metrics.csv",index=False)
    selective_classes = []
    for scope,current,column in [("Dobra exportada: uma janela",default,"eligible_candidate"),
                                 ("Dobra exportada: 2/3 contíguo",default_subset,"confirmed_candidate")]:
        truth = current.label.map({c:i for i,c in enumerate(classes)}).fillna(-1).to_numpy().astype(int)
        emitted = current[column].to_numpy()
        for c,name in enumerate(classes):
            true_count = int(np.sum(truth==c))
            emitted_count = int(np.sum(emitted==c))
            correct_count = int(np.sum((emitted==c)&(truth==c)))
            selective_classes.append(dict(scope=scope,species=name,positive_endpoints=true_count,
                true_groups=current.loc[current.label.eq(name),"group"].nunique(),
                emitted_as_species=emitted_count,correct_identifications=correct_count,
                emitted_species_precision=correct_count/emitted_count if emitted_count else None,
                correctly_identified_fraction=correct_count/true_count if true_count else None))
    pd.DataFrame(selective_classes).to_csv(OUT/"selective_per_class.csv",index=False)
    atomic_json(OUT/"protocol.json",dict(version=VERSION,feature_signature=identity,classes=classes,
        source_splits=split_records,exports=exports,
        description="Three outer folds; source/content-disjoint. 20 known species; weak recording labels.",
        selection="Inner source/class-weighted accuracy on supported classes; singleton classes stay in training.",
        calibration="Temperature and abstention threshold fitted on positive calibration sources only, balanced by class/source.",
        default_export_fold=default_fold,physical_board_tested=False,
        default_selection="First fold meeting the calibration target, using no outer test result; otherwise all-reject fold 0.",
        temporal_note="Corpus endpoints are 1 s apart; firmware windows 0.992 s. Gaps/files reset state. Quality guards not measured.",
        limitations=["No local Arduino microphone recordings or external field validation.",
                     "Scores do not guarantee per-prediction correctness or identify unknown species.",
                     "Only two noise sources provide contiguous endpoints; not a false-alarms/hour estimate.",
                     "Aggregated source accuracy differs from accuracy of a single sound/window."]))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print(pd.DataFrame(operating).to_string(index=False),flush=True)


if __name__=="__main__":
    main()
