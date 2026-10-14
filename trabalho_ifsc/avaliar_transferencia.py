"""Reavalia os 17 arquivos públicos JÁ examinados em 09/10; teste retrospectivo."""
from pathlib import Path
import json
import hashlib
import pandas as pd
from experimento_cnn import ROOT, infer

OUT = ROOT / 'results/ifsc_cnn_2026_10_11'

def main():
    result = OUT / 'transfer_recordings.csv'
    if result.exists():
        raise FileExistsError('Avaliação preservada; não sobrescrever.')
    checkpoint = OUT / 'cnn.pt'
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    files = pd.read_csv(ROOT / 'results/species_nested_2026_10_09/external_recording_predictions.csv')
    files = files.drop_duplicates('path').sort_values('path')
    assert len(files) == 17
    training = pd.read_csv(OUT / 'partitions.csv')
    assert not set(files.path) & set(training.path)
    rows = []
    for row in files.itertuples():
        predicted = infer(ROOT / row.path, checkpoint)
        rows.append(dict(path=row.path, group=row.group, truth=row.label,
                         prediction=predicted['top3'][0]['species'],
                         windows=predicted['windows'], correct=predicted['top3'][0]['species'] == row.label))
    assert before == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    pd.DataFrame(rows).to_csv(result, index=False)
    metric = dict(recordings=len(rows), correct=sum(x['correct'] for x in rows),
                  accuracy=sum(x['correct'] for x in rows)/len(rows),
                  classes=5, independent_new_acquisition=False,
                  historical_audio_used_in_previous_research=True,
                  model_frozen_unchanged=True, checkpoint_sha256=before,
                  absence_noise_not_tested=True,
                  biological_independence_unknown=True)
    (OUT / 'transfer_metrics.json').write_text(json.dumps(metric, ensure_ascii=False, indent=2))
    print(json.dumps(metric, indent=2), flush=True)

if __name__ == '__main__':
    main()
