"""Replay every held-out prediction and audit grouped training partitions."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

import train_pattern_classifiers as s
from pattern_classifiers import assert_partitions, require
from torres_lstm import digest, predict_probabilities


def resolve_document_revision(path=None):
    if path is not None:
        return json.loads(Path(path).read_text())
    for candidate in [
        s.ROOT / 'data/metadata/document_revision.json',
        s.ROOT / 'output/pattern_slides_2026_10_16/document_revision.json',
    ]:
        if candidate.is_file():
            return json.loads(candidate.read_text())
    return None


def verify_protected_files(root, expected, revision=None):
    """Permit only an explicitly recorded teaching-notebook revision, never weights."""
    changed=[]
    for path,sha in expected.items():
        current=digest(root/path)
        if current==sha:continue
        if path!='notebooks/detector_ternario_treinamento_teste_aplicacao.ipynb':
            raise ValueError(f'Protected scientific artifact changed: {path}')
        entry=(revision or {}).get('files',{}).get(path,{})
        if (revision or {}).get('kind')!='authorized_document_revision' or entry.get('original_sha256')!=sha or entry.get('current_sha256')!=current:
            raise ValueError(f'Document revision requires an explicit matching receipt: {path}')
        changed.append(path)
    return changed


def verify_run(out, reference, sequences, document_revision=None, device=None):
    out=Path(out);reference=Path(reference);sequences=Path(sequences)
    torch.set_num_threads(4)
    if device is None:
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    protocol=json.loads((out/'protocol.json').read_text())
    frame=pd.read_csv(out/'segments.csv.gz',low_memory=False)
    require(digest(reference/'segments.csv.gz')==protocol['segments_sha256'],'Reference segments SHA-256 mismatch.')
    require(digest(reference/'classical_features.npy')==protocol['features_sha256'],'Reference features SHA-256 mismatch.')
    cache=json.loads((reference/'feature_cache.json').read_text())
    require(digest(sequences)==cache['sha256'],'Sequence cache SHA-256 mismatch.')
    source_dir=out/'source_snapshot' if (out/'source_snapshot').is_dir() else s.ROOT
    for path,sha in protocol['code_sha256'].items():
        require(digest(source_dir/path)==sha,f'Frozen code changed: {path}')
    revision=resolve_document_revision(document_revision)
    revised_documents=verify_protected_files(s.ROOT,protocol['protected_sha256'],revision)
    raw=np.load(sequences,mmap_mode='r');stats=np.load(reference/'classical_features.npy',mmap_mode='r')
    expected_groups=int(frame.group.nunique())
    expected_aedes=int(frame.groupby('group').true_label.first().eq(2).sum())
    checks=[];all_groups={}
    for name in s.NAMES:
        collected=[]
        for fold in range(5):
            d=out/f'outer_{fold}'/name;require((d/'complete.json').exists(),f'Missing complete.json for {name} fold {fold}')
            parts=pd.read_csv(d/'partitions.csv.gz',low_memory=False)
            fit,val,test,dev=s.partitions(frame,fold)
            for indices,role in [(fit,'fit'),(val,'validation'),(test,'test')]:
                require(np.array_equal(np.flatnonzero(parts.role.eq(role)),indices),f'Partition role mismatch for {name} fold {fold}: {role}')
            assert_partitions(frame,fit,val,test)
            frozen=json.loads((d/'freeze.json').read_text())
            require(frozen['before_test_inference'] and frozen['fit_rows']==dev.tolist(),f'Invalid freeze receipt for {name} fold {fold}')
            require(digest(d/'selection.json')==frozen['selection_sha256'],f'Selection SHA-256 mismatch for {name} fold {fold}')
            selection=json.loads((d/'selection.json').read_text())
            if name in s.NAMES[:3]:
                require(digest(d/'model.joblib')==frozen['model_sha256'],f'Model SHA-256 mismatch for {name} fold {fold}')
                bundle=joblib.load(d/'model.joblib')
                np.testing.assert_allclose(bundle['scaler'].mean_,stats[dev].mean(0,dtype=np.float64))
                p=s.ml_proba(bundle['model'],bundle['scaler'].transform(stats[test]));rule=bundle['rule']
                require(selection['validation_score']==max(c['score'] for c in selection['candidates']),f'Validation score mismatch for {name} fold {fold}')
            else:
                require(digest(d/'model.pt')==frozen['model_sha256'],f'Model SHA-256 mismatch for {name} fold {fold}')
                require(digest(d/'scaler.npz')==frozen['scaler_sha256'],f'Scaler SHA-256 mismatch for {name} fold {fold}')
                features=raw if name=='cnn' else stats;axes=(0,1) if name=='cnn' else 0
                for indices,path in [(fit,'inner_scaler.npz'),(dev,'scaler.npz')]:
                    z=np.load(d/path)
                    np.testing.assert_allclose(z['mean'],features[indices].mean(axes,keepdims=True))
                    np.testing.assert_allclose(z['std'],features[indices].std(axes,keepdims=True).clip(.001))
                z=np.load(d/'scaler.npz');x=((features[test]-z['mean'])/z['std']).astype('float32')
                checkpoint=torch.load(d/'model.pt',map_location='cpu',weights_only=True)
                model=s.nn_factory(name).to(device);model.load_state_dict(checkpoint['state_dict'])
                p=predict_probabilities(model,x,device,batch_size=256);rule=checkpoint['rule']
                history=json.loads((d/'inner_history.json').read_text())
                chosen=history[selection['best_epoch']-1]['validation_loss']
                require(chosen<=min(r['validation_loss'] for r in history)+1e-4,f'Inner epoch selection mismatch for {name} fold {fold}')
                require(len(json.loads((d/'refit_history.json').read_text()))==selection['best_epoch'],f'Refit history length mismatch for {name} fold {fold}')
                del model,x
            np.testing.assert_allclose(p,np.load(d/'window_test_probabilities.npy'),atol=1e-6,rtol=1e-5)
            groups=s.table(frame.iloc[test],p)
            groups['argmax_prediction']=groups[[f'prob_{c}' for c in range(3)]].to_numpy().argmax(1)
            groups['calibrated_prediction']=s.apply_policy(groups,rule)
            replay=groups.set_index('group').sort_index()
            actual=pd.read_csv(d/'group_predictions.csv').set_index('group').sort_index()
            require(replay.index.equals(actual.index),f'Group index mismatch for {name} fold {fold}')
            for column in ['true_label','argmax_prediction','calibrated_prediction']:
                np.testing.assert_array_equal(replay[column],actual[column])
            collected.append(actual.reset_index())
            checks.append(dict(model=name,fold=fold,replayed=True,groups=len(groups),aedes=int(groups.true_label.eq(2).sum())))
            print('VERIFIED',name,fold,flush=True)
        combined=pd.concat(collected,ignore_index=True)
        require(len(combined)==combined.group.nunique()==expected_groups,f'Combined group coverage mismatch for {name}')
        require(int(combined.true_label.eq(2).sum())==expected_aedes,f'Combined Aedes count mismatch for {name}')
        all_groups[name]=combined
    choices=json.loads((out/'model_selection.json').read_text())
    selected=pd.read_csv(out/'selected_oof.csv').set_index('group').sort_index()
    expected=[]
    for item in choices:
        name=max(s.NAMES,key=lambda n:item['validation_scores'][n]);require(name==item['selected_model'],'Selected model mismatch.')
        expected.append(all_groups[name][all_groups[name].fold.eq(item['fold'])])
    expected=pd.concat(expected).set_index('group').sort_index()
    np.testing.assert_array_equal(selected.argmax_prediction,expected.argmax_prediction)
    final_manifest_path=out/'final'/'manifest.json'
    if final_manifest_path.is_file():
        final_manifest=json.loads(final_manifest_path.read_text())
        model_file='model.pt' if final_manifest['model'] in ['cnn','mlp'] else 'model.joblib'
        require(digest(out/'final'/model_file)==final_manifest['model_sha256'],'Final model SHA-256 mismatch.')
        if final_manifest['model'] in ['cnn','mlp']:
            require(digest(out/'final'/'scaler.npz')==final_manifest['scaler_sha256'],'Final scaler SHA-256 mismatch.')
    target='verification_after_document_update.json' if revision else 'verification.json'
    payload=dict(complete=True,contexts=checks,group_coverage_per_model=expected_groups,
        validation_only_model_selection=True,protected_artifacts_unchanged=not revised_documents,
        protected_model_artifacts_unchanged=True,authorized_document_revisions=revised_documents)
    target_path=out/target
    if not (target_path.is_file() and json.loads(target_path.read_text())==payload):
        s.write(target_path,payload)
    return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--sequences',type=Path,required=True)
    parser.add_argument('--document-revision',type=Path,help='Explicit receipt for the authorized notebook update; never permits model/data changes.')
    args=parser.parse_args()
    verify_run(args.out,args.reference,args.sequences,args.document_revision)


if __name__=='__main__':main()
