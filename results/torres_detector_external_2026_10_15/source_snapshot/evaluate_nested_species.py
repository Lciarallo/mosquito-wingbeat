"""Post-freeze acquisition and species evaluation; no tuning from predictions."""
from pathlib import Path
import argparse
import hashlib
import json
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import fetch_humbug
from nested_species_ensemble import ROOT, NAMES, now, audio_probabilities, file_hash, atomic_json

TAXA={1:'Aedes aegypti',2:'Aedes albopictus',3:'Culex quinquefasciatus',
      4:'Anopheles gambiae',5:'Anopheles arabiensis',6:'Anopheles dirus',
      7:'Culex pipiens',8:'Anopheles minimus',9:'Anopheles stephensi'}


def pcm_keys(path):
    """Exact decoded identity and aligned fragments at three declared sizes.

    Includes gain-normalised variants, excluding near-constant/silent fragments.
    Not a search for arbitrary offsets, edits, resampling implementations or
    biological identity. The audit is deliberately bounded to known files.
    """
    wave,sr=sf.read(path,dtype='float32',always_2d=True)
    wave=wave.mean(1)
    if sr!=8000:
        d=np.gcd(sr,8000);wave=resample_poly(wave,8000//d,sr//d)
    keys=set()
    pieces=[wave]
    for size in [8000,7936,2400]:
        pieces.extend(wave[i:i+size] for i in range(0,len(wave)-size+1,size))
    for fragment in pieces:
        if not len(fragment):continue
        for a in [fragment,fragment/ max(float(np.abs(fragment).max()),1e-12)*.9]:
            pcm=np.rint(np.clip(a,-1,32767/32768)*32768).astype('<i2')
            if np.ptp(pcm.astype('int32'))<=2:continue
            keys.add((len(pcm),hashlib.sha256(pcm.tobytes()).hexdigest()))
    return keys


def verify_freeze(out):
    freeze=json.loads((out/'freeze.json').read_text())
    for p,h in freeze['artifacts_sha256'].items():
        if file_hash(ROOT/p)!=h:raise ValueError('Frozen artifact changed: '+p)
    if file_hash(ROOT/'nested_species_ensemble.py')!=freeze['inference_code_sha256']:
        raise ValueError('Inference code changed after freeze.')
    return freeze


def acquire(out):
    freeze=verify_freeze(out)
    planpath=out/'fresh_selection.json'
    if planpath.exists():raise FileExistsError('Post-freeze selection already exists; never silently reselect.')
    centralpath=ROOT/'tmp/reproductions_2026_10_01/cdmsc_central_full.json'
    central=json.loads(centralpath.read_text())
    if central.get('partial_central_directory'):raise ValueError('Incomplete ZIP index.')
    dest=ROOT/'data/external/species_nested_fresh_2026_10_09'
    dest.mkdir(parents=True,exist_ok=True)
    old_files=sorted({p.resolve() for root in ['data/raw','data/external','data/noise','data/external_datasets','tmp']
                      for p in (ROOT/root).rglob('*.wav') if dest not in p.parents})
    seen_names={p.name for p in old_files}
    old_files.append(ROOT/'drive_audio.wav')
    pools={}
    for sid in TAXA:
        pools[sid]=sorted([e for e in central['entries'] if Path(e['name']).name.startswith(f'S_{sid}_D_3_')
                          and e['name'].endswith('.wav') and Path(e['name']).name not in seen_names],
                         key=lambda e:int(Path(e['name']).stem.split('_')[-1]))
    plan=dict(selection_utc=now(),frozen_utc=freeze['frozen_utc'],source='https://zenodo.org/records/19095788',
        archive=central['file'],central_directory_sha256=file_hash(centralpath),
        rule='For each of 9 species: first 4 unused numeric D3 file IDs whose decoded duration >=1s. Scan at most 30 unused IDs per species; document short clips. No prediction-based selection or replacement after novelty audit.',
        candidates={str(sid):pool[:30] for sid,pool in pools.items()}, files=[],
        familiar_corpus_not_new_campaign=True,original_individual_session_identity_unavailable=True,
        frozen_artifacts_sha256=freeze['artifacts_sha256'],
        acquisition_code_sha256=file_hash(Path(__file__)), known_files=len(old_files))
    if plan['selection_utc']<=freeze['frozen_utc']:raise ValueError('Selection must follow freeze.')
    atomic_json(planpath,plan)
    # Snapshot old identities before acquiring any new audio or running inference.
    byte_hashes=set();pcm_hashes=set();errors=[];inventory=[]
    for i,path in enumerate(old_files):
        digest=file_hash(path);byte_hashes.add(digest)
        inventory.append(dict(path=str(path.relative_to(ROOT)),sha256=digest))
        try:pcm_hashes.update(pcm_keys(path))
        except (sf.LibsndfileError,ValueError) as error:errors.append(dict(path=str(path),error=str(error)))
        if i%1000==0:print('NOVELTY INDEX',i,'/',len(old_files),flush=True)
    pd.DataFrame(inventory).to_csv(out/'known_audio_inventory.csv',index=False)
    plan['novelty_index']=dict(byte_hashes=len(byte_hashes),pcm_signatures=len(pcm_hashes),
        errors=errors,scope='Known decoded WAVs; whole PCM and aligned 1s/.992s/.3s fragments at 8kHz, raw and peak-normalised. Does not establish biological ancestry or arbitrary-offset uniqueness.')
    atomic_json(planpath,plan)
    def one_species(sid):
        chosen=[];excluded=[]
        for e in plan['candidates'][str(sid)]:
            start=e['offset'];end=start+30+len(e['name'])+e['compressed']+1024
            raw=fetch_humbug.byte_range(plan['archive']['links']['self'],start,end)
            header=struct.unpack_from('<4s5H3I2H',raw)
            if header[0]!=b'PK\x03\x04':raise ValueError('Invalid ZIP local header.')
            off=30+header[-2]+header[-1];data=raw[off:off+e['compressed']]
            if e['method']==8:data=zlib.decompress(data,-15)
            elif e['method']!=0:raise ValueError('Unsupported ZIP method.')
            if len(data)!=e['uncompressed'] or zlib.crc32(data)!=e['crc']:raise ValueError('ZIP CRC/size mismatch.')
            path=dest/Path(e['name']).name;path.write_bytes(data)
            duration=sf.info(path).duration
            item=dict(path=str(path.relative_to(ROOT)),sha256=file_hash(path),reference_species=TAXA[sid],
                      seconds=duration,crc32=e['crc'],entry=e['name'],group='BioDCASE_D3|'+path.stem)
            if duration<1:
                excluded.append(dict(**item,reason='duration<1s'))
                continue
            chosen.append(item)
            print('POST-FREEZE FILE',path.name,duration,flush=True)
            if len(chosen)>=4:break
        return chosen,excluded
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(one_species,sorted(TAXA)))
    plan['files']=[item for items,_ in results for item in items]
    plan['duration_exclusions']=[item for _,items in results for item in items]
    audit=[];accepted_hashes=set();accepted_pcm=set()
    for item in plan['files']:
        keys=pcm_keys(ROOT/item['path'])
        matches=len(keys & pcm_hashes);within_matches=len(keys & accepted_pcm)
        byte_duplicate=item['sha256'] in byte_hashes or item['sha256'] in accepted_hashes
        excluded=byte_duplicate or matches>0 or within_matches>0
        audit.append(dict(path=item['path'],reference_species=item['reference_species'],
                          byte_duplicate=byte_duplicate,known_pcm_matches=matches,
                          fresh_pcm_matches=within_matches,excluded=excluded))
        # Even excluded files count in the within-selection duplicate audit.
        accepted_hashes.add(item['sha256']);accepted_pcm.update(keys)
    pd.DataFrame(audit).to_csv(out/'novelty_audit.csv',index=False)
    plan['download_complete_utc']=now()
    plan['accepted_files_after_audit']=sum(not x['excluded'] for x in audit)
    plan['by_species_selected']={sp:sum(x['reference_species']==sp for x in plan['files']) for sp in TAXA.values()}
    atomic_json(planpath,plan);verify_freeze(out)
    print('ACQUISITION COMPLETE',len(plan['files']),'selected,',plan['accepted_files_after_audit'],'novel',flush=True)


def evaluate(out):
    from research_metrics_v2 import classification_metrics
    freeze=verify_freeze(out)
    plan=json.loads((out/'fresh_selection.json').read_text())
    if plan['selection_utc']<=freeze['frozen_utc']:raise ValueError('Selection chronology invalid.')
    if (out/'external_metrics.csv').exists():raise FileExistsError('External evaluation already exists.')
    bundle=joblib.load(out/'models/final_bundle.joblib')
    classes=np.asarray(bundle['classes'])
    audit=pd.read_csv(out/'novelty_audit.csv')
    excluded=set(audit[audit.excluded].path)
    windows=[];recordings=[]
    for item in plan['files']:
        if item['path'] in excluded:continue
        if file_hash(ROOT/item['path'])!=item['sha256']:raise ValueError('Fresh audio changed.')
        scores,quality,info=audio_probabilities(ROOT/item['path'],bundle)
        for name,p in scores.items():
            # Invalid windows stay in the denominator; never manufacture a
            # correct negative or remove them to improve classification scores.
            true=item['reference_species'];pred=classes[p.argmax(1)]
            for i in range(len(p)):
                windows.append(dict(model=name,path=item['path'],group=item['group'],label=true,
                    start_s=i,quality_ok=bool(quality[i]),forced_species=str(pred[i]),score=float(p[i].max())))
            rec=p.mean(0);candidate=str(classes[rec.argmax()])
            accepted=bool(quality.all() and rec.max()>=bundle['thresholds'][name]['threshold'])
            recordings.append(dict(model=name,path=item['path'],group=item['group'],label=true,
                forced_species=candidate,quality_ok=bool(quality.all()),accepted=accepted,
                score=float(rec.max()),identification=candidate if accepted else 'UNCERTAIN',**info))
        print('SPECIES INFERENCE',Path(item['path']).name,flush=True)
    w=pd.DataFrame(windows);r=pd.DataFrame(recordings)
    if w.empty:raise ValueError('No qualifying novel external audio; do not replace based on predictions.')
    w.to_csv(out/'external_window_predictions.csv',index=False)
    r.to_csv(out/'external_recording_predictions.csv',index=False)
    rows=[];per=[]
    for unit,frame in [('window',w),('recording',r)]:
        for name,sub in frame.groupby('model'):
            correct=sub.label.eq(sub.forced_species)&sub.quality_ok
            accepted=sub.accepted if unit=='recording' else sub.quality_ok
            known=sorted(sub.label.unique())
            predicted=np.where(sub.quality_ok,sub.forced_species,'INVALID')
            rows.append(dict(model=name,unit=unit,n=len(sub),files=sub.group.nunique(),correct=int(correct.sum()),
                accuracy=float(correct.mean()),balanced_accuracy=float(balanced_accuracy_score(sub.label,predicted)),
                macro_f1=float(f1_score(sub.label,predicted,labels=known,average='macro',zero_division=0)),
                **{k:v for k,v in classification_metrics(sub.label,predicted).items()
                   if k.startswith('macro_f1_') or k=='predicted_taxa_outside_support'},
                coverage=float(accepted.mean()),accepted_accuracy=float(correct[accepted].mean()) if accepted.any() else None,
                correct_identifications=int((correct&accepted).sum()) if unit=='recording' else None))
            for species in known:
                a=sub[sub.label.eq(species)];recall=(a.label.eq(a.forced_species)&a.quality_ok).mean()
                emit=sub[sub.forced_species.eq(species)&sub.quality_ok]
                precision=emit.label.eq(species).mean() if len(emit) else None
                per.append(dict(model=name,unit=unit,species=species,n=len(a),files=a.group.nunique(),
                    forced_recall=float(recall),forced_precision=float(precision) if precision is not None else None))
    pd.DataFrame(rows).to_csv(out/'external_metrics.csv',index=False)
    pd.DataFrame(per).to_csv(out/'external_per_species.csv',index=False)
    # Previously examined user audio is only a retrospective positive control.
    scores,quality,info=audio_probabilities(ROOT/'drive_audio.wav',bundle)
    private=[]
    for name,p in scores.items():
        candidate=str(classes[p.mean(0).argmax()])
        accepted=bool(quality.all() and p.mean(0).max()>=bundle['thresholds'][name]['threshold'])
        private.append(dict(model=name,windows=len(p),forced_correct_windows=int(((classes[p.argmax(1)]=='Aedes aegypti')&quality).sum()),
            invalid_windows=int((~quality).sum()),recording_candidate=candidate,
            recording_identification=candidate if accepted else 'UNCERTAIN',
            reference='User-reported species throughout recording; not independently annotated events.',**info))
    pd.DataFrame(private).to_csv(out/'private_audio_metrics.csv',index=False)
    verify_freeze(out)
    atomic_json(out/'external_evaluation_audit.json',dict(evaluated_utc=now(),models_thresholds_unchanged=True,
        external_selection_after_freeze=True,unknown_biological_ancestry=True,
        code_sha256=file_hash(Path(__file__)),no_padding_repetition=True,
        all_invalid_windows_retained=True,noise_presence_validated=False))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print(pd.DataFrame(private).to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/species_nested_2026_10_09')
    parser.add_argument('--phase',choices=['acquire','evaluate'],required=True)
    args=parser.parse_args()
    (acquire if args.phase=='acquire' else evaluate)(args.output.resolve())


if __name__=='__main__':main()
