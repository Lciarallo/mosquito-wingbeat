"""Frozen external diagnostics with author weights, distinct from paper replication.

The original model consumes 300ms mono audio. The explicitly declared common
992ms comparison averages its eight-class outputs from three real subwindows.
No training, gain adjustment, denoising, threshold search or fold selection.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import librosa
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/faithful_reproduction_2026_10_03/mosquitosong"
META = ROOT / "results/faithful_reproduction_2026_10_03/common_test_metadata.json"
MODELS = ROOT / "data/external/faithful_2026_10_03/mosquitosong_models"


def original_jitter(start):
    np.random.seed(start)
    return int(np.random.randint(120))


def summaries(y, predicted, quality, margin):
    known = quality.astype(bool)
    cm = np.array([[int(((~predicted)&known&(y==v)).sum()),
                    int((predicted&known&(y==v)).sum()),
                    int(((~known)&(y==v)).sum())] for v in [0,1]])
    tn, fp, fn, tp = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
    npos, nneg = int((y==1).sum()), int((y==0).sum())
    recall = tp/npos if npos else None
    specificity = tn/nneg if nneg else None
    return dict(n=len(y), positive_windows=npos, negative_windows=nneg,
                invalid=int((~known).sum()), confusion=cm.tolist(),
                tp=tp, tn=tn, fp=fp, fn=fn, recall=recall, specificity=specificity,
                precision=tp/(tp+fp) if tp+fp else None,
                balanced_accuracy=(recall+specificity)/2 if recall is not None and specificity is not None else None,
                roc_auc=float(roc_auc_score(y, margin)) if npos and nneg else None,
                positive_only=nneg==0)


def main():
    plan = json.loads((OUT.parent / "external_protocol.json").read_text())
    for filename, expected in plan["model_weights_sha256"].items():
        assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest() == expected
    metadata = json.loads(META.read_text())
    native_waves, common_waves = [], []
    native_y, common_y, common_q, native_scope, common_scope = [], [], [], [], []
    native_group, common_group = [], []
    for record in metadata:
        x, sr = librosa.load(str(ROOT/record["path"]), sr=8000)
        assert sr == 8000
        if record["size_16k"] != 15872:
            raise ValueError("Common comparison requires original full 992ms intervals")
        expected = record["windows"]
        assert abs(len(x)//7936-expected) <= 1
        for start in range(0, len(x), 1200):
            shift = original_jitter(start)
            y = x[start+shift:start+shift+2400]
            if len(y) == 2400:
                native_waves.append(y)
                native_y.append(int(record["reference_aedes"]))
                native_scope.append(record["scope"])
                native_group.append(record["group"])
        for i in range(expected):
            base = i*7936
            for start in [0,2400,4800]:
                shift = original_jitter(start)
                y = x[base+start+shift:base+start+shift+2400]
                assert len(y) == 2400, "No padding/repetition of waveform allowed"
                common_waves.append(y)
            common_y.append(int(record["reference_aedes"]))
            common_q.append(record["quality_ok"][i])
            common_scope.append(record["scope"])
            common_group.append(record["group"])
    native_x = np.asarray(native_waves, dtype=np.float32)[:,:,None]
    common_x = np.asarray(common_waves, dtype=np.float32)[:,:,None]
    native_y = np.asarray(native_y)
    common_y = np.asarray(common_y)
    common_q = np.asarray(common_q, dtype=bool)
    native_scope, common_scope = np.asarray(native_scope), np.asarray(common_scope)
    rows = []
    all_common_predictions = []
    for fold in range(1,11):
        model = tf.keras.models.load_model(str(MODELS/("cnn_model_min_val_round%d.h5" % fold)))
        for kind, x, y, q, scopes, groups in [
            ("native_300ms", native_x, native_y, np.ones(len(native_y),bool), native_scope,native_group),
            ("common_992ms_three_subwindows", common_x, common_y, common_q, common_scope,common_group)]:
            prob = model.predict(x, batch_size=64, verbose=0)
            if kind.startswith("common"):
                prob = prob.reshape(-1,3,8).mean(1)
                all_common_predictions.append(prob)
            # Sorted eight labels verified against the original evaluation.
            class_index = np.argmax(prob, axis=1)
            predicted = class_index < 2
            margin = prob[:,:2].max(1)-prob[:,2:].max(1)
            for scope in sorted(set(scopes)):
                mask = scopes==scope
                row = dict(fold=fold, scope=scope, protocol=kind,
                           **summaries(y[mask], predicted[mask], q[mask], margin[mask]))
                rows.append(row)
                print("EXTERNAL",fold,kind,scope,"TP",row["tp"],"FN",row["fn"],"TN",row["tn"],"FP",row["fp"], flush=True)
            np.savez_compressed(OUT/("external_fold%d_%s.npz" % (fold,kind)),
                                y=y, p=predicted, probabilities=prob, margin=margin,
                                quality=q, scope=scopes, group=np.asarray(groups))
        (OUT/"external_metrics.json").write_text(json.dumps(rows,indent=2))
        tf.keras.backend.clear_session()
    # The average across author folds is an additional frozen ensemble, not the
    # original cross-validation or an independently trained binary classifier.
    prob = np.mean(all_common_predictions,axis=0)
    predicted = np.argmax(prob,axis=1) < 2
    margin = prob[:,:2].max(1)-prob[:,2:].max(1)
    for scope in sorted(set(common_scope)):
        mask=common_scope==scope
        rows.append(dict(fold="all_folds_mean_derived", scope=scope,
                         protocol="common_992ms_three_subwindows",
                         **summaries(common_y[mask],predicted[mask],common_q[mask],margin[mask])))
    np.savez_compressed(OUT/"external_derived_ensemble.npz",y=common_y,p=predicted,
                        probabilities=prob,margin=margin,quality=common_q,
                        scope=common_scope,group=np.asarray(common_group))
    (OUT/"external_metrics.json").write_text(json.dumps(rows,indent=2))
    (OUT/"external_execution.json").write_text(json.dumps(dict(status="complete", training=False,
         weights_unchanged=True, private_raw_audio_exported=False,
         native_windows=len(native_y), common_windows=len(common_y),
         comparison_extension=True, native_protocol_distinct_from_paper=True),indent=2))


if __name__ == "__main__":
    main()
