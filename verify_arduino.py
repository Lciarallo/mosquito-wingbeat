"""Validate complete PCM->features->score C++ parity and causal confirmation."""
from pathlib import Path
import ctypes
import json
import subprocess
import numpy as np
import pandas as pd
import joblib
import arduino_frontend as frontend
from improve_device import ROOT, atomic_json, file_hash, SEED
from train_arduino import OUT, calibrated

def main():
    temporary=ROOT/"tmp/arduino-verify";temporary.mkdir(parents=True,exist_ok=True)
    library=temporary/"frontend.so"
    command=["g++","-std=c++11","-O2","-Wall","-Wextra","-Werror","-shared","-fPIC",
             str(ROOT/"firmware/host_bridge.cpp"),"-o",str(library)]
    subprocess.run(command,check=True)
    lib=ctypes.CDLL(str(library)); fp=np.ctypeslib.ndpointer(dtype=np.float32,ndim=1,flags="C_CONTIGUOUS")
    pcm_pointer=np.ctypeslib.ndpointer(dtype=np.int16,ndim=1,flags="C_CONTIGUOUS")
    lib.mosquito_features.argtypes=[pcm_pointer,ctypes.c_size_t,fp];lib.mosquito_features.restype=ctypes.c_int
    lib.mosquito_score.argtypes=[fp];lib.mosquito_score.restype=ctypes.c_float
    lib.mosquito_frontend_bytes.restype=ctypes.c_size_t
    protocol=json.loads((OUT/"protocol.json").read_text());candidate=protocol["source_splits"][0]["selected_candidate"]
    bundle=joblib.load(OUT/"models"/f"candidate{candidate}_fold0.joblib")["bundle"]
    segments=pd.read_csv(ROOT/"results/segments.csv");waves=np.load(ROOT/"data/cache/waves.npy",mmap_mode="r")
    rng=np.random.default_rng(SEED)
    test=np.flatnonzero(segments.fold_species.to_numpy()==0)
    noise=test[segments.iloc[test].label.eq("noise").to_numpy()]
    positive=test[segments.iloc[test].label.ne("noise").to_numpy()]
    # All held-out negative windows plus a random positive sample; no training inputs.
    sampled=np.r_[noise,rng.choice(positive,400,replace=False)]
    pcm=frontend.pcm16(waves[sampled]);expected=frontend.features(pcm)
    actual=[];actual_scores=[]
    for row in pcm:
        output=np.zeros(68,dtype="float32")
        assert lib.mosquito_features(row[:frontend.WINDOW_SAMPLES].copy(),frontend.WINDOW_SAMPLES,output)==1
        actual.append(output);actual_scores.append(lib.mosquito_score(output))
    actual=np.stack(actual);actual_scores=np.array(actual_scores)
    reference=calibrated(bundle,expected);error=np.abs(actual-expected)
    same=(actual_scores>=bundle["threshold_10"])==(reference>=bundle["threshold_10"])
    # Full pipeline can cross thresholds for nearly tied tree features. Report
    # this explicitly instead of silently loosening the prediction assertion.
    assert error.max()<.005,(error.max(),np.unravel_index(error.argmax(),error.shape))
    ip=np.ctypeslib.ndpointer(dtype=np.int32,ndim=1,flags="C_CONTIGUOUS")
    lib.mosquito_confirm.argtypes=[ip,ctypes.c_size_t,ip]
    decisions=np.array([0,1,1,0,0,1,1,1,0,0],dtype="int32");out=np.empty_like(decisions)
    lib.mosquito_confirm(decisions,len(decisions),out)
    expected_votes,eligible=frontend.persistence(decisions,["a"]*len(decisions),np.arange(len(decisions)),.5)
    assert np.array_equal(out.astype(bool),expected_votes)
    # A time gap and a new file must each reset the causal vote.
    votes,_=frontend.persistence([1,1,1,1,1,1],["a"]*5+["b"],[0,1,2,5,6,0],.5)
    assert np.array_equal(votes,[0,0,1,0,0,0])
    silence=np.zeros(frontend.WINDOW_SAMPLES,dtype="int16");silent=np.zeros(68,dtype="float32")
    assert lib.mosquito_features(silence,len(silence),silent)==1 and np.isfinite(silent).all()
    assert np.allclose(silent,frontend.features(silence)[0],atol=1e-5)
    # Gain and DC-invariance away from quantization floors are prerequisites
    # for using RMS-normalized corpus audio with raw microphone PCM.
    changed=pcm[0].astype("float32")*.4+100
    gain_error=float(np.max(np.abs(frontend.features(changed,replay_pcm=False)[0]-expected[0])))
    assert gain_error<.001
    audit=dict(native_compiler_command=[str(arg).replace(str(ROOT)+"/","") for arg in command],
        tested_windows=len(sampled),heldout_positive_windows=400,heldout_noise_windows=len(noise),
        max_absolute_feature_error=float(error.max()),mean_absolute_feature_error=float(error.mean()),
        max_absolute_score_error=float(np.max(np.abs(actual_scores-reference))),
        matching_decisions=int(same.sum()),decision_agreement=float(same.mean()),
        tested_operating_mode="Calibração FPR 10% (default firmware)",
        streaming_frontend_object_bytes=int(lib.mosquito_frontend_bytes()),
        gain_and_dc_feature_max_error=gain_error,confirmation_and_gap_reset_tests_passed=True,
        actual_board_flashing=False,real_time_deadline_measured_on_board=False,
        artifacts={str(p.relative_to(ROOT)):file_hash(p) for p in (ROOT/"firmware").rglob("*") if p.is_file()})
    atomic_json(OUT/"native_audit.json",audit);print(json.dumps(audit,indent=2,ensure_ascii=False))

if __name__=="__main__": main()
