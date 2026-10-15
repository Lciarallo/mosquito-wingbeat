"""Acquire a fixed unused acoustic test only after model freeze."""
import json, struct, zlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import soundfile as sf
from train_cross_domain_strategies import OUT, ROOT
from train_reproduction import now, sha
import fetch_humbug as fetch

D = ROOT / 'data/external/cross_domain_strategy_test_2026_10_03'


def main():
    frozen = json.loads((OUT / 'freeze.json').read_text())
    for p, h in frozen['models_sha256'].items(): assert sha(ROOT / p) == h
    D.mkdir(exist_ok=True)
    planpath = OUT / 'fresh_selection.json'
    if planpath.exists():
        plan = json.loads(planpath.read_text())
        if plan.get('completed_utc'): raise FileExistsError('Completed test may not be reselected.')
    else:
        central = json.loads((ROOT / 'tmp/reproductions_2026_10_01/cdmsc_central_full.json').read_text())
        assert not central['partial_central_directory']
        seen = {p.name for r in [ROOT / 'data/external', ROOT / 'tmp'] for p in r.rglob('*.wav') if D not in p.parents}
        entries = []
        for sid, taxon, count in [(1, 'Aedes aegypti', 16), (2, 'Aedes albopictus', 4),
                                  (7, 'Culex pipiens', 4), (8, 'Anopheles minimus', 4),
                                  (9, 'Anopheles stephensi', 4)]:
            pool = sorted([e for e in central['entries'] if Path(e['name']).name.startswith(f'S_{sid}_D_3_')
                           and e['name'].endswith('.wav') and Path(e['name']).name not in seen],
                          key=lambda e: int(Path(e['name']).stem.split('_')[-1]))
            assert len(pool) >= count
            entries.extend(dict(entry=e, reference_species=taxon, reference_aedes=sid == 1) for e in pool[:count])
        plan = dict(selection_utc=now(), frozen_utc=frozen['frozen_utc'], models_sha256=frozen['models_sha256'],
                    source='https://zenodo.org/records/19095788', archive=central['file'], planned=entries,
                    rule='First unused numeric IDs:16 class1,4 each class2/7/8/9, all acoustic D3; no model inference before selection.',
                    no_optical_d5=True, recording_session_identity_unavailable=True, files=[])
        assert plan['selection_utc'] > frozen['frozen_utc']; planpath.write_text(json.dumps(plan, indent=2))
    known = set()
    for manifest in (ROOT / 'results').rglob('*selection.json'):
        if manifest == planpath: continue
        try:
            obj = json.loads(manifest.read_text())
            known.update(item['sha256'] for item in obj.get('files', []) if 'sha256' in item)
        except (ValueError, TypeError): pass
    known.update(__import__('pandas').read_csv(ROOT / 'results/reliability_2026_10_01/segments.csv', low_memory=False).content_sha256.dropna())
    def one(item):
        e = item['entry']; path = D / Path(e['name']).name
        if not path.exists():
            raw = fetch.byte_range(plan['archive']['links']['self'], e['offset'], e['offset'] + 30 + len(e['name']) + e['compressed'] + 1024)
            h = struct.unpack_from('<4s5H3I2H', raw); assert h[0] == b'PK\x03\x04'
            start = 30 + h[-2] + h[-1]; data = raw[start:start + e['compressed']]
            data = zlib.decompress(data, -15) if e['method'] == 8 else data
            assert len(data) == e['uncompressed'] and zlib.crc32(data) == e['crc']; path.write_bytes(data)
        digest = sha(path); assert digest not in known, 'Exact previous file; never replace based on prediction.'
        return dict(path=str(path.relative_to(ROOT)), sha256=digest, group='CDMSC_D3|' + path.stem,
                    source='CDMSC_D3', seconds=sf.info(path).duration,
                    reference_species=item['reference_species'], reference_aedes=item['reference_aedes'])
    with ThreadPoolExecutor(max_workers=3) as pool:
        for item in pool.map(one, plan['planned']):
            plan['files'].append(item); print('FRESH', len(plan['files']), item['path'], flush=True)
    plan['completed_utc'] = now(); planpath.write_text(json.dumps(plan, indent=2))


if __name__ == '__main__': main()
