"""Offline experimental classification using the frozen notebook-transfer models.

Predictions are hypotheses; a binary negative means other species/noise, not
absence of mosquito. Does not alter the neural or Arduino frontend.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from reproduce_notebook_transfer import ROOT, OUT, digest, verify_freeze, audio_bag, predict_bundle


def verified_bundle(task='species', variant=None, family=None, output=OUT):
    verify_freeze(output)
    protocol=json.loads((output/'protocol.json').read_text())
    # The feature functions are loaded from the received notebook; its original
    # SHA is recorded inside the frozen protocol and checked before inference.
    reference='external/notebook_reference_2026_10_14/Mosquito_v2.ipynb'
    if digest(ROOT/reference)!=protocol['inputs_sha256'][reference]:
        raise ValueError('Received reference feature source changed.')
    selected=json.loads((output/'selected_models.json').read_text())[task]
    variant=variant or selected['variant'];family=family or selected['family']
    return joblib.load(output/'models'/f'final_{task}_{variant}_{family}.joblib')


def predict_audio(path, task='species', variant=None, family=None):
    bundle=verified_bundle(task,variant,family)
    bags,windows=audio_bag(Path(path));settings=bundle['settings'];variant=settings['variant']
    p=predict_bundle(bundle,np.array([bags[variant]]))[0];classes=np.array(settings['classes'])
    predicted=str(classes[p.argmax()]);accepted=bool(p.max()>=settings['rejection_threshold'])
    result=dict(audio=str(path),task=task,variant=variant,family=settings['family'],candidate=predicted,accepted=accepted,
                emitted_label=predicted if accepted else 'ABSTAIN',confidence=float(p.max()),
                probabilities={c:float(v) for c,v in zip(classes,p)},complete_windows=len(windows[variant]),
                experimental=True,biological_confirmation=False,calibration_unit='recording feature bag',
                temporal_ground_truth_available=False)
    if task=='binary':
        score=float(p[np.searchsorted(classes,'Aedes aegypti')]);result.update(aegypti_score=score,aegypti_cutoff=settings['aegypti_cutoff'],aegypti_detected=score>=settings['aegypti_cutoff'],
         negative_label_meaning='Other mosquito species or noise; not absence of mosquito')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--audio',required=True);parser.add_argument('--task',choices=['binary','species'],default='species');parser.add_argument('--variant',choices=['notebook61','v4_308','v4_lfcc334']);parser.add_argument('--family',choices=['lr','tree','rf','knn']);parser.add_argument('--output-summary');args=parser.parse_args()
    result=predict_audio(args.audio,args.task,args.variant,args.family);text=json.dumps(result,ensure_ascii=False,indent=2);print(text)
    if args.output_summary:
        output=Path(args.output_summary)
        if output.exists():raise FileExistsError('Do not overwrite previous predictions.')
        output.parent.mkdir(parents=True,exist_ok=True);output.write_text(text+'\n')
