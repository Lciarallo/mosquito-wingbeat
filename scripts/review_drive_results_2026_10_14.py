"""Read-only replay of received Drive claims; no training or parameter selection."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
WORK = ROOT / 'output/drive_review_2026_10_14_161041'
OUT = ROOT / 'results/drive_review_2026_10_14_161041'
from research_metrics_v2 import classification_metrics, bootstrap_metrics


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def load_pipeline(version):
    """Load each unchanged source with its own unchanged harmonic module."""
    directory = ROOT / 'scripts' if version == 'current' else WORK / 'local_before/scripts'
    robust = types.ModuleType('robust_audio_features')
    robust.__file__ = str(ROOT / 'scripts/robust_audio_features.py')
    exec(compile((directory / 'robust_audio_features.py').read_bytes(), str(directory / 'robust_audio_features.py'), 'exec'), robust.__dict__)
    sys.modules['robust_audio_features'] = robust
    module = types.ModuleType('review_pipeline_' + version)
    # Resolve the original project models identically for both source versions.
    module.__file__ = str(ROOT / 'scripts/predict_audio.py')
    exec(compile((directory / 'predict_audio.py').read_bytes(), str(directory / 'predict_audio.py'), 'exec'), module.__dict__)
    return module


def one_row(name, path, truth, version, df, summary):
    predicted = summary['primary_species'] or 'NO_CANDIDATE'
    genus = summary['primary_genus'] or 'NO_CANDIDATE'
    emission = summary['confirmed_species'] or 'REJECTED'
    genus_emission = summary['confirmed_genus'] or 'REJECTED'
    return dict(
        file=name, path=path, reference_species=truth, reference_genus=truth.split()[0], version=version,
        candidate_species=predicted, candidate_genus=genus, emitted_species=emission, emitted_genus=genus_emission,
        correct_species=bool(predicted == truth), correct_genus=bool(genus == truth.split()[0]),
        accepted_species=bool(summary['confirmed_species']), accepted_genus=bool(summary['confirmed_genus']),
        total_windows=len(df), detected_windows=int(df.detected.sum()), swarm_windows=int(df.is_aedes_swarm.sum()),
        confirmed_species_windows=int(df.confirmed_species.notna().sum()),
        candidate_aegypti_windows=int((df.candidate_species == 'Aedes aegypti').sum()),
        median_f0_hz=summary['median_flight_f0_hz'], input_sha256=digest(ROOT / path),
    )


def metric_row(sub, scope, task, version):
    truth = sub['reference_' + task].to_numpy()
    candidates = sub['candidate_' + task].to_numpy()
    emissions = sub['emitted_' + task].to_numpy()
    accepted = sub['accepted_' + task].to_numpy(bool)
    m = classification_metrics(truth, candidates, accepted)
    # Actual accepted output can differ from the mode of all candidates.
    m['accepted_accuracy'] = float((emissions[accepted] == truth[accepted]).mean()) if accepted.any() else None
    m['correct_emissions_over_all'] = float(((emissions == truth) & accepted).mean())
    m['accepted_output_is_candidate_mode'] = bool(np.array_equal(emissions[accepted], candidates[accepted]))
    m['scope'], m['task'], m['version'] = scope, task, version
    return m


def probe_batch_dependency(module, summary):
    """Keep one real Drive window fixed and change only other feature rows."""
    raw, _ = module.load_audio_via_ffmpeg(ROOT / 'data/raw/drive_test/drive_audio.wav', target_sr=16000)
    pcm = module.arduino_frontend.pcm16(np.asarray(raw, np.float32)).ravel()
    target = pcm[int(50 * 16000):int(50 * 16000) + module.WINDOW_SAMPLES]
    feat = module.arduino_frontend.features(np.stack([target]), replay_pcm=True)[0]
    pair_low = np.stack([feat, feat]); pair_high = pair_low.copy()
    pair_low[1, [31, 46, 32, 58]] = 0.0
    pair_high[1, [31, 46, 32, 58]] = 4.0
    a = module.HierarchicalMosquitoPipeline(region='global')
    b = module.HierarchicalMosquitoPipeline(region='global')
    captures = []
    for model, pair in [(a, pair_low), (b, pair_high)]:
        original = model.predict_species_probabilities
        predictions = []
        def capture(x, flutter_info=None):
            p, energy = original(x, flutter_info)
            predictions.append(p.copy())
            return p, energy
        model.predict_species_probabilities = capture
        rows = model.evaluate_windows(pair, [target.copy(), target.copy()])
        captures.append((predictions[0], rows[0]))
    pa, ra = captures[0]; pb, rb = captures[1]
    return dict(
        fixed_window_start_s=50.0, identical_first_feature_row=bool(np.array_equal(pair_low[0], pair_high[0])),
        changed_only_other_rows=True, changed_other_feature_indices=[31, 46, 32, 58],
        first_prediction_bit_identical=bool(np.array_equal(pa, pb)),
        max_absolute_probability_difference=float(np.max(np.abs(pa - pb))),
        low_context_candidate=ra['candidate_species'], high_context_candidate=rb['candidate_species'],
        low_context_probabilities=pa.tolist(), high_context_probabilities=pb.tolist(),
        nature='Controlled feature-matrix intervention, not an alternative physical recording.',
        implication='Recording-level feature means make this script transductive across windows, even with a past-only timeline.',
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audit = json.loads((OUT / 'received_multiclass_28files_v5_audit.json').read_text())
    paths = json.loads((OUT / 'audit_file_paths.json').read_text())['resolved']
    species_map = {'Classe_S1': 'Aedes aegypti', 'Classe_S2': 'Aedes albopictus',
                   'Classe_S7': 'Culex pipiens', 'Classe_S8': 'Anopheles minimus', 'Classe_S9': 'Anopheles stephensi'}
    inputs = {r['file']: dict(path=paths[r['file']], truth=species_map[r['true_class']]) for r in audit['per_file_results_global_mode']}
    plan = json.loads((ROOT / 'results/species_nested_v4_2026_10_14/external_plan.json').read_text())
    paired_names = []
    for r in plan['paired_files']:
        name = Path(r['path']).name; paired_names.append(name)
        assert digest(ROOT / r['path']) == r['sha256']
        inputs[name] = dict(path=r['path'], truth=r['reference_species'])
    frozen_paths = [ROOT / 'scripts/predict_audio.py', ROOT / 'scripts/robust_audio_features.py', Path(__file__),
                    ROOT / 'arduino_frontend.py', ROOT / 'arduino_mic_emulator.py', ROOT / 'train_hierarchical_models.py']
    frozen_paths += sorted((ROOT / 'results/arduino_species/models').glob('species_fold*.npz'))
    frozen_paths += list((WORK / 'local_before/scripts').glob('*.py'))
    frozen_paths += [ROOT / p['path'] for p in inputs.values()]
    before = {p.relative_to(ROOT).as_posix(): digest(p) for p in frozen_paths}
    save(OUT / 'review_protocol.json', dict(
        fixed_before_replay_utc=now(), files_sha256=before, selection_or_retuning=False,
        kind='Retrospective read-only diagnostic replay; not a v5 model training freeze.',
        inputs=inputs, audit_scope_files=list(paths), paired_17_files=paired_names, versions=['previous', 'current'],
        regional_prior='global for both public comparisons', analysis_window_s=.992, hop_s=.5,
        candidate_recording_rule='primary_species/genus: most frequent candidate over detected windows',
        emission_recording_rule='confirmed_species/genus: most frequent past-persistence output over windows',
        limitations=['All files previously available; no blind acquisition', 'Five species only',
                     'Private species label supplied by user', 'No temporal event annotations or negative recordings',
                     'Genus persistence is not a separately calibrated confidence threshold'],
    ))
    cli_outputs = {}
    for mode, region, emulated in [('emulated_brazil', 'brazil', True), ('raw_brazil', 'brazil', False), ('raw_global', 'global', False)]:
        csv = OUT / (mode + '_predictions.csv'); js = OUT / (mode + '_summary.json')
        command = [sys.executable, str(ROOT / 'scripts/predict_audio.py'), '--audio',
                   'data/raw/drive_test/drive_audio.wav', '--region', region, '--output-csv', str(csv), '--output-summary', str(js)]
        if emulated: command.append('--emulate-arduino')
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, env={**os.environ, 'PYTHONPATH': str(ROOT)})
        (OUT / ('terminal_' + mode + '.txt')).write_text('Command: ' + ' '.join(command) + '\n' + result.stdout + result.stderr + '\nExit code: ' + str(result.returncode) + '\n')
        if result.returncode: raise RuntimeError('CLI replay failed: ' + mode)
        cli_outputs[mode] = json.loads(js.read_text())
        print('CLI', mode, cli_outputs[mode]['detected_windows'], cli_outputs[mode]['species_distribution'], flush=True)
    records = []
    for version in ['previous', 'current']:
        module = load_pipeline(version)
        classifier = module.HierarchicalMosquitoPipeline(region='global')
        for name, item in sorted(inputs.items()):
            if version == 'current' and name == 'drive_audio.wav':
                df = pd.read_csv(OUT / 'raw_global_predictions.csv'); summary = cli_outputs['raw_global']
            else:
                df, summary = module.process_audio_file(ROOT / item['path'], classifier=classifier)
                directory = OUT / 'per_file' / version; directory.mkdir(parents=True, exist_ok=True)
                df.to_csv(directory / (Path(name).stem + '.csv'), index=False)
                save(directory / (Path(name).stem + '_summary.json'), summary)
            row = one_row(name, item['path'], item['truth'], version, df, summary)
            records.append(row)
            print('REPLAY', version, name, row['candidate_species'], 'correct=' + str(row['correct_species']), flush=True)
        if version == 'current': save(OUT / 'batch_inductivity_probe.json', probe_batch_dependency(module, cli_outputs['raw_global']))
    frame = pd.DataFrame(records); frame.to_csv(OUT / 'recording_predictions.csv', index=False)
    metrics = []; intervals = {}
    for scope, names in [('received_28', list(paths)), ('paired_17', paired_names), ('public_27', [n for n in paths if n != 'drive_audio.wav'])]:
        chosen = frame[frame.file.isin(names)]
        for task in ['species', 'genus']:
            predictions = {}; acceptances = {}
            for version in ['previous', 'current']:
                sub = chosen[chosen.version == version].sort_values('file')
                metrics.append(metric_row(sub, scope, task, version))
                predictions[version] = sub['candidate_' + task].to_numpy()
                # Separate bootstrap of actual emissions, so candidate/emission mismatches cannot inflate accuracy.
                acceptances[version] = np.ones(len(sub), bool)
            intervals[scope + '_' + task + '_candidates'] = bootstrap_metrics(sub['reference_' + task].to_numpy(), predictions, acceptances, seed=20261014, clusters=sub.file.to_numpy())
            emissions = {}; emit_accept = {}
            for version in ['previous', 'current']:
                sub = chosen[chosen.version == version].sort_values('file')
                emissions[version] = sub['emitted_' + task].to_numpy()
                emit_accept[version] = sub['accepted_' + task].to_numpy(bool)
            intervals[scope + '_' + task + '_emissions'] = bootstrap_metrics(sub['reference_' + task].to_numpy(), emissions, emit_accept, seed=20261014, clusters=sub.file.to_numpy())
    pd.DataFrame(metrics).to_csv(OUT / 'metrics.csv', index=False)
    save(OUT / 'metrics.json', metrics); save(OUT / 'bootstrap_intervals.json', intervals)
    mismatches = []
    current = frame[frame.version == 'current'].set_index('file')
    for r in audit['per_file_results_global_mode']:
        ours = current.loc[r['file']]
        if ours.candidate_species != species_map[r['predicted_class']] or abs(ours.median_f0_hz - r['median_f0_hz']) > 0.11:
            mismatches.append(dict(file=r['file'], received_prediction=species_map[r['predicted_class']], replay_prediction=ours.candidate_species,
                                   received_f0=r['median_f0_hz'], replay_f0=ours.median_f0_hz))
    changed = [p for p, h in before.items() if digest(ROOT / p) != h]
    save(OUT / 'replay_verification.json', dict(
        completed_utc=now(), read_only_frozen_inputs_unchanged=not changed, changed_paths=changed,
        received_audit_prediction_or_f0_mismatches=mismatches, scope_metrics=metrics,
        nested_v5_trained_artifacts_available=(ROOT / 'results/species_nested_v5_2026_10_14/freeze.json').exists(),
        received_dryad_63_62_claim='Not reproduced: no received v5 OOF predictions or trained/frozen v5 artifacts.',
        private_raw_audio_uploaded=False, parameters_retuned=False,
    ))
    assert not changed
    for m in metrics: print('METRIC', json.dumps(m, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
