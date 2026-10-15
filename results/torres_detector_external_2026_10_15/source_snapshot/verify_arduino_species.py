"""Verify exported C++ against Python on held-out source groups and real PCM."""
import ctypes
import json
import subprocess
import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax,expit
import arduino_frontend as frontend
from arduino_species_models import probabilities,confirm_species
from improve_device import ROOT,SEED,atomic_json,file_hash
from train_arduino import calibrated
from train_arduino_species import OUT


def main():
    protocol = json.loads((OUT/"protocol.json").read_text())
    fold = protocol["default_export_fold"]
    source = protocol["source_splits"][fold]
    bundle = joblib.load(OUT/"models"/f"candidate{source['selected_candidate']}_fold{fold}.joblib")["bundle"]
    presence = joblib.load(ROOT/"results/arduino/models"/f"candidate{source['presence_candidate']}_fold{fold}.joblib")["bundle"]
    temporary = ROOT/"tmp/arduino-species-verify"
    temporary.mkdir(parents=True,exist_ok=True)
    library = temporary/"species.so"
    command = ["g++","-std=c++11","-O2","-Wall","-Wextra","-Werror","-shared","-fPIC",
               str(ROOT/"firmware/species_host_bridge.cpp"),"-o",str(library)]
    subprocess.run(command,check=True)
    lib = ctypes.CDLL(str(library))
    f1 = np.ctypeslib.ndpointer(dtype=np.float32,ndim=1,flags="C_CONTIGUOUS")
    f2 = np.ctypeslib.ndpointer(dtype=np.float32,ndim=2,flags="C_CONTIGUOUS")
    i1 = np.ctypeslib.ndpointer(dtype=np.int32,ndim=1,flags="C_CONTIGUOUS")
    pcm_pointer = np.ctypeslib.ndpointer(dtype=np.int16,ndim=1,flags="C_CONTIGUOUS")
    lib.mosquito_species_features.argtypes = [pcm_pointer,ctypes.c_size_t,f1]
    lib.mosquito_species_features.restype = ctypes.c_int
    lib.mosquito_species_batch.argtypes = [f2,ctypes.c_size_t,f2,f1,i1]
    lib.mosquito_species_confirm.argtypes = [i1,i1,ctypes.c_size_t,i1]
    lib.mosquito_species_frontend_bytes.restype = ctypes.c_size_t
    assert lib.mosquito_species_fold()==fold

    def native(x):
        x = np.ascontiguousarray(x,dtype="float32")
        p = np.empty((len(x),20),dtype="float32")
        presence_p = np.empty(len(x),dtype="float32")
        labels = np.empty(len(x),dtype="int32")
        lib.mosquito_species_batch(x,len(x),p,presence_p,labels)
        return p,presence_p,labels

    def reference(x):
        p,gate = probabilities(bundle,x),calibrated(presence,x)
        labels = np.where((p.max(axis=1)>=bundle["confidence_threshold"]) &
                          (gate>=presence["threshold_10"]),p.argmax(axis=1),-1)
        return p,gate,labels

    segments = pd.read_csv(ROOT/"results/segments.csv")
    x = np.load(ROOT/"data/cache/arduino_features.npy")
    test = np.flatnonzero(segments.fold_species.to_numpy()==fold)
    expected,expected_gate,expected_labels = reference(x[test])
    actual,actual_gate,actual_labels = native(x[test])
    assert np.isfinite(actual).all() and np.allclose(actual.sum(axis=1),1,atol=1e-5)
    assert np.array_equal(actual.argmax(axis=1),expected.argmax(axis=1))
    assert np.array_equal(actual_labels,expected_labels)
    assert np.max(np.abs(actual-expected))<.001
    assert np.max(np.abs(actual_gate-expected_gate))<.001

    # Numeric checkpoints include all layers, calibration and the presence gate.
    with np.load(OUT/"models"/f"species_fold{fold}.npz") as saved:
        value = x[test].copy()
        for i in range(len(bundle["weights"])):
            value = value@saved[f"weights{i}"].T+saved[f"bias{i}"]
            if i<len(bundle["weights"])-1:
                value = np.maximum(0,value)
        restored = softmax(value/saved["temperature"],axis=1)
        hidden = np.maximum(0,x[test]@saved["presence_weights0"].T+saved["presence_bias0"])
        restored_gate = expit(saved["presence_calibration_slope"]*(hidden@saved["presence_weights1"]+
                              saved["presence_bias1"])+saved["presence_calibration_bias"])
        assert np.allclose(restored,expected,atol=1e-6)
        assert np.allclose(restored_gate,expected_gate,atol=1e-4)
        assert saved["classes"].tolist()==protocol["classes"] and saved["fold"]==fold

    waves = np.load(ROOT/"data/cache/waves.npy",mmap_mode="r")
    noise = test[segments.iloc[test].label.eq("noise").to_numpy()]
    positive = test[segments.iloc[test].label.ne("noise").to_numpy()]
    sampled = np.r_[noise,np.random.default_rng(SEED).choice(positive,400,replace=False)]
    pcm = frontend.pcm16(waves[sampled])
    expected_features = frontend.features(pcm)
    actual_features = []
    for row in pcm:
        output = np.zeros(68,dtype="float32")
        assert lib.mosquito_species_features(row[:frontend.WINDOW_SAMPLES].copy(),frontend.WINDOW_SAMPLES,output)==1
        actual_features.append(output)
    actual_features = np.stack(actual_features)
    pcm_actual,pcm_gate,pcm_labels = native(actual_features)
    pcm_expected,pcm_expected_gate,pcm_expected_labels = reference(expected_features)
    assert np.max(np.abs(actual_features-expected_features))<.005
    assert np.array_equal(pcm_actual.argmax(axis=1),pcm_expected.argmax(axis=1))
    assert np.array_equal(pcm_labels,pcm_expected_labels)
    assert np.max(np.abs(pcm_actual-pcm_expected))<.001

    # Warmup, alternating species, an uncertain current window and explicit
    # capture reset. Expected outcomes specify the policy independently.
    labels = np.array([2,2,2,3,2,-1,2,4,4,4,4,4],dtype="int32")
    resets = np.array([0,0,0,0,0,0,0,0,0,1,0,0],dtype="int32")
    voted = np.empty_like(labels)
    lib.mosquito_species_confirm(labels,resets,len(labels),voted)
    assert np.array_equal(voted,[-1,-1,2,-1,2,-1,2,-1,4,-1,-1,4])
    reference_vote,_ = confirm_species([2,2,2,2,2,2],["a"]*5+["b"],[0,1,2,5,6,0])
    assert np.array_equal(reference_vote,[-1,-1,2,-1,-1,-1])
    ordered = segments.iloc[test].copy().sort_values(["path","start_s"])
    positions = pd.Series(np.arange(len(test)),index=test).loc[ordered.index].to_numpy()
    labels = actual_labels[positions].astype("int32")
    resets = np.r_[True,ordered.path.to_numpy()[1:]!=ordered.path.to_numpy()[:-1]]
    resets |= np.r_[True,np.abs(np.diff(ordered.start_s.to_numpy())-1.)>1e-5]
    voted = np.empty_like(labels)
    lib.mosquito_species_confirm(labels,resets.astype("int32"),len(labels),voted)
    reference_vote,_ = confirm_species(labels,ordered.path,ordered.start_s)
    assert np.array_equal(voted,reference_vote)

    audit = dict(default_fold=fold,feature_only_test_windows=len(test),
        feature_only_matching_argmax=int(np.sum(actual.argmax(axis=1)==expected.argmax(axis=1))),
        feature_only_matching_threshold_decisions=int(np.sum(actual_labels==expected_labels)),
        feature_only_max_probability_error=float(np.max(np.abs(actual-expected))),
        feature_only_max_presence_error=float(np.max(np.abs(actual_gate-expected_gate))),
        complete_pcm_test_windows=len(sampled),complete_pcm_positive_windows=400,complete_pcm_noise_windows=len(noise),
        complete_pcm_matching_argmax=int(np.sum(pcm_actual.argmax(axis=1)==pcm_expected.argmax(axis=1))),
        complete_pcm_matching_threshold_decisions=int(np.sum(pcm_labels==pcm_expected_labels)),
        complete_pcm_max_feature_error=float(np.max(np.abs(actual_features-expected_features))),
        complete_pcm_max_probability_error=float(np.max(np.abs(pcm_actual-pcm_expected))),
        complete_pcm_max_presence_error=float(np.max(np.abs(pcm_gate-pcm_expected_gate))),
        temporal_matching_decisions=len(test),temporal_warmup_rejection_alternation_reset_passed=True,
        complete_numeric_checkpoint_restoration_passed=True,
        streaming_frontend_object_bytes=int(lib.mosquito_species_frontend_bytes()),
        actual_board_flashing=False,real_time_deadline_measured_on_board=False,
        native_compiler_command=[a.replace(str(ROOT)+"/","") for a in command],
        artifacts={str(p.relative_to(ROOT)):file_hash(p) for p in (ROOT/"firmware/MosquitoSpecies").glob("*") if p.is_file()})
    atomic_json(OUT/"native_audit.json",audit)
    print(json.dumps(audit,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
