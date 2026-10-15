"""Replay every neural outer test prediction and verify frozen training identities."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_ternary_lstm import ROOT, GroupPool, read_json
from torres_lstm import TorresLSTM, digest, predict_probabilities, write_json
from ternary_protocol import check_partition, apply_policy
from ternary_reporting import evaluate_predictions


def verify(run,device,output=None):
    run=Path(run)
    assert read_json(run/"complete.json")["complete"]
    protocol=read_json(run/"protocol.json")
    frozen=read_json(run/"freeze.json")
    for source,expected in protocol["source_sha256"].items():
        assert digest(run/"source"/source)==expected
    frame=pd.read_csv(run/"segments.csv.gz",low_memory=False)
    assert len(frame)==protocol["windows"]
    assert frame.group.nunique()==protocol["groups"]
    assert frame.pcm_block.nunique()==protocol["pcm_blocks"]
    assert frame.groupby("pcm_block").outer_fold.nunique().max()==1
    assert not frame.domain.astype(str).str.lower().eq("mosquitosong").any()
    row_positions=pd.Series(np.arange(len(frame)),index=frame.row_id)
    raw_path=ROOT/protocol["inputs"]["features"]
    assert digest(raw_path)==protocol["inputs"]["feature_sha256"]
    raw=np.load(raw_path,mmap_mode="r")
    metrics=read_json(run/"metrics.json")
    output={}
    for name in ["lstm","bilstm"]:
        saved=np.load(run/f"{name}_window_oof.npy",allow_pickle=False)
        assert np.isfinite(saved).all()
        largest=0.
        for context in [f"outer_{k}" for k in range(5)]+["final"]:
            directory=run/context/name
            freeze=read_json(directory/"freeze.json")
            assert digest(directory/"freeze.json")==frozen["context_freeze_sha256"][f"{context}/{name}"]
            assert digest(directory/"model.pt")==freeze["checkpoint_sha256"]
            assert digest(directory/"scaler.npz")==freeze["scaler_sha256"]
            assert digest(directory/"selection.json")==freeze["selection_sha256"]
            fit=row_positions.loc[freeze["fit_rows"]].to_numpy(int)
            selection=read_json(directory/"selection.json")
            inner=row_positions.loc[selection["inner_fit_rows"]].to_numpy(int)
            val=row_positions.loc[selection["validation_rows"]].to_numpy(int)
            assert set(inner)|set(val)==set(fit)
            check_partition(frame,inner,val)
            if context!="final":
                fold=int(context.split("_")[1])
                test=np.flatnonzero(frame.outer_fold.eq(fold))
                check_partition(frame,fit,test)
                assert set(fit)==set(np.flatnonzero(frame.outer_fold.ne(fold)))
            else:
                assert set(fit)==set(range(len(frame)))
            for file,indices in [("scaler.npz",fit),("inner_scaler.npz",inner)]:
                with np.load(directory/file,allow_pickle=False) as scale:
                    np.testing.assert_allclose(scale["mean"],raw[indices].mean((0,1),keepdims=True),rtol=1e-5)
                    np.testing.assert_allclose(scale["std"],raw[indices].std((0,1),keepdims=True).clip(.001),rtol=1e-5)
            history=read_json(directory/"inner_history.json")
            epoch=next(r for r in history if r["epoch"]==freeze["best_epoch"])
            assert abs(epoch["validation_group_loss"]-freeze["validation_loss"])<1e-12
            assert len(read_json(directory/"refit_history.json"))==freeze["best_epoch"]
            if context!="final":
                with np.load(directory/"scaler.npz",allow_pickle=False) as scale:
                    x=((raw[test]-scale["mean"])/scale["std"]).astype("float32")
                checkpoint=torch.load(directory/"model.pt",map_location="cpu",weights_only=True)
                model=TorresLSTM(**checkpoint["config"]).to(device)
                model.load_state_dict(checkpoint["state_dict"])
                actual=predict_probabilities(model,x,device,batch_size=512)
                largest=max(largest,float(np.max(np.abs(actual-saved[test]))))
                np.testing.assert_allclose(actual,saved[test],rtol=1e-5,atol=1e-6)
        group=GroupPool(frame).table(saved)
        prediction=pd.read_csv(run/f"{name}_event_oof.csv")
        np.testing.assert_array_equal(group.group,prediction.group)
        np.testing.assert_array_equal(group.true_label,prediction.true_label)
        recalibrated=np.empty(len(group),dtype=int)
        group_folds=frame.groupby("group").outer_fold.first().reindex(group.group).to_numpy(int)
        for fold in range(5):
            mask=group_folds==fold
            rule=read_json(run/f"outer_{fold}"/name/"freeze.json")["policy"]
            recalibrated[mask]=apply_policy(group.loc[mask],rule)
        np.testing.assert_array_equal(recalibrated,prediction.calibrated_prediction)
        for decision,guess in [("argmax",group[[f"prob_{c}" for c in range(3)]].to_numpy().argmax(1)),("calibrated",recalibrated)]:
            expected=next(r for r in metrics["neural_models"] if r["model"]==name and r["decision"]==decision)
            actual=evaluate_predictions(group.true_label,guess)
            assert actual["confusion_matrix_raw"]==expected["confusion_matrix_raw"]
        output[name]=dict(group_oof_coverage=int(frame.group.nunique()),outer_models_replayed=5,max_probability_replay_difference=largest)
    assert frozen["selected_final_neural_model"]==min(frozen["final_candidates"],key=lambda x:x["validation_loss"])["model"]
    result=dict(complete=True,models=output,normalizers_verified=24,source_snapshot_verified=True,
        identical_reference_population=True,outer_PCM_disjoint=True,selection_without_outer_test=True,
        reference_accuracy=metrics["reference"]["accuracy"])
    if output is not None:
        output = Path(output)
        if output.exists():
            raise FileExistsError("Existing verification is preserved: " + str(output))
        write_json(output,result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    p.add_argument("--output",type=Path,help="Optional new receipt; existing files are never overwritten.")
    a=p.parse_args()
    torch.set_num_threads(4)
    device=torch.device(a.device if a.device!="auto" else "cuda" if torch.cuda.is_available() else "cpu")
    import json
    print(json.dumps(verify(a.run,device,a.output),indent=2))
