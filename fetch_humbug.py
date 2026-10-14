"""Download the official HumBugDB archives, checking published MD5s."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import http.client
import struct
import zlib
import hashlib
import json
import urllib.request
import urllib.error
import time
import numpy as np
import pandas as pd

ALIASES = {
    'ae aegypti': 'Aedes aegypti', 'ae albopictus': 'Aedes albopictus',
    'an arabiensis': 'Anopheles arabiensis', 'an dirus': 'Anopheles dirus',
    'an gambiae': 'Anopheles gambiae', 'an gambiae ss': 'Anopheles gambiae',
    'an minimus': 'Anopheles minimus', 'an stephensi': 'Anopheles stephensi',
    'culex quinquefasciatus': 'Culex quinquefasciatus', 'culex tarsalis': 'Culex tarsalis',
}
# Complex-level annotations are deliberately not converted into exact species.


def byte_range(url, start, end):
    request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                assert response.status == 206
                assert response.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
                data = response.read()
            assert len(data) == end - start + 1
            return data
        except (urllib.error.URLError, http.client.IncompleteRead, TimeoutError) as error:
            if attempt == 4:
                raise
            delay = min(30, 2 ** (attempt + 1))
            if isinstance(error, urllib.error.HTTPError) and error.code == 429:
                delay = min(60, max(delay, int(error.headers.get('Retry-After', '30'))))
            print('Retrying incomplete/failed range', start, 'in', delay, 'seconds', flush=True)
            time.sleep(delay)


def entries(item):
    marker = OUT / (item['key'] + '.entries.json')
    if marker.exists():
        return json.loads(marker.read_text())
    tail = byte_range(item['links']['self'], item['size'] - 65536, item['size'] - 1)
    eocd = tail.rfind(b'PK\x05\x06')
    _, _, _, _, count, length, offset, _ = struct.unpack_from('<4s4H2IH', tail, eocd)
    data = byte_range(item['links']['self'], offset, offset + length - 1)
    position, result = 0, []
    while position < len(data):
        fields = struct.unpack_from('<4s6H3I5H2I', data, position)
        assert fields[0] == b'PK\x01\x02'
        name_len, extra, comment = fields[10:13]
        name = data[position + 46:position + 46 + name_len].decode()
        result.append(dict(name=name, compressed=fields[8], uncompressed=fields[9],
                           offset=fields[16], method=fields[4], crc=fields[7]))
        position += 46 + name_len + extra + comment
    assert len(result) == count
    marker.write_text(json.dumps(result))
    return result


def selected_download(record):
    """A bounded, class-balanced sample chosen without examining predictions."""
    metadata_item = next(item for item in record['files'] if item['key'].endswith('.csv'))
    fetch(metadata_item)
    metadata = pd.read_csv(OUT / metadata_item['key'])
    metadata['year'] = pd.to_datetime(metadata.record_datetime, dayfirst=True,
                                    format='mixed', errors='coerce').dt.year
    # Excluding collections <=2017 also avoids potentially shared CDC recordings.
    metadata = metadata[metadata.year.ge(2018) & metadata.length.ge(.992) &
                        metadata.sound_type.isin(['mosquito', 'background'])].copy()
    metadata['label'] = metadata.species.map(ALIASES)
    metadata.loc[metadata.sound_type.eq('background'), 'label'] = 'noise'
    metadata['label'] = metadata.label.fillna('unknown_mosquito')
    metadata['group'] = metadata[['country', 'record_datetime', 'name']].fillna('').agg('|'.join, axis=1)
    groups = metadata.groupby('group', sort=True)
    # Keep all segments from one original recording in one split. For each group,
    # download its longest labelled excerpt; this bounds transfer and source bias.
    representatives = metadata.loc[groups.length.idxmax()].copy()
    representatives['stratum'] = representatives.group.map(groups.label.apply(
        lambda labels: next((value for value in labels if value not in ['noise', 'unknown_mosquito']),
                            'unknown_mosquito' if 'unknown_mosquito' in labels.values else 'noise')))
    rng = np.random.default_rng(20260930)
    selected = []
    for label, part in representatives.groupby('stratum', sort=True):
        cap = 60 if label == 'noise' else 30 if label == 'unknown_mosquito' else 40
        selected.extend(rng.permutation(part.index)[:cap].tolist())
    selected_groups = representatives.loc[selected, 'group']
    grouped_metadata = metadata[metadata.group.isin(selected_groups)]
    # Retain one longest excerpt for EACH label in an original recording.
    # Otherwise a long background excerpt could displace its mosquito excerpt.
    chosen = grouped_metadata.loc[grouped_metadata.groupby(['group', 'label']).length.idxmax()].copy()
    # Previously examined examples become adaptation data, never new evidence.
    for identifier in [220292, 222024]:
        row = metadata[metadata.id.eq(identifier)]
        chosen = pd.concat([chosen, row], ignore_index=True)
    chosen = chosen.drop_duplicates('id').sort_values('id')
    index = {}
    for item in sorted(record['files'], key=lambda item: item['key']):
        if item['key'].endswith('.zip'):
            for entry in entries(item):
                index[entry['name']] = (item, entry)
    chosen.to_csv(OUT / 'selected_metadata.csv', index=False)
    (OUT / 'audio').mkdir(exist_ok=True)

    def download_row(row):
        item, entry = index[f'{row.id}.wav']
        target = OUT / 'audio' / entry['name']
        if target.exists() and target.stat().st_size == entry['uncompressed'] and zlib.crc32(target.read_bytes()) == entry['crc']:
            data = target.read_bytes()
        else:
            start = entry['offset']
            # WAV entry names are ten bytes; local headers have no expected extra,
            # but read a margin and honour the actual extra length.
            end = start + 30 + len(entry['name']) + entry['compressed'] + 1024
            content = byte_range(item['links']['self'], start, end)
            header = struct.unpack_from('<4s5H3I2H', content)
            assert header[0] == b'PK\x03\x04' and entry['method'] == 8
            offset = 30 + header[-2] + header[-1]
            data = zlib.decompress(content[offset:offset + entry['compressed']], -15)
            assert len(data) == entry['uncompressed'] and zlib.crc32(data) == entry['crc']
            target.write_bytes(data)
        print('Verified excerpt', row.id, row.label, flush=True)
        return dict(id=int(row.id), archive=item['key'], entry=entry['name'],
                    crc32=entry['crc'], sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))

    with ThreadPoolExecutor(max_workers=2) as pool:
        manifest = list(pool.map(download_row, chosen.itertuples(index=False)))
    (OUT / 'selected_manifest.json').write_text(json.dumps(dict(
        selection_seed=20260930, recording_year_minimum=2018,
        max_groups_per_known_class=40, max_noise_groups=60, max_unknown_groups=30,
        selection_before_inference=True, integrity='ZIP entry CRC32 and local SHA256; full archive MD5 not checked',
        metadata_md5=digest(OUT / metadata_item['key']), files=manifest), indent=2))
    print('Selected corpus complete:', len(manifest), 'excerpts', flush=True)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data/external/humbug'


def digest(path):
    h = hashlib.md5()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fetch(item):
    path = OUT / item['key']
    expected = item['checksum'].split(':')[1]
    if path.exists() and path.stat().st_size == item['size'] and digest(path) == expected:
        print('Verified cached', path.name, flush=True)
        return
    temporary = path.with_suffix(path.suffix + '.part')
    # A stale partial download is replaced; only verified files become final.
    print('Downloading', path.name, item['size'], flush=True)
    with urllib.request.urlopen(item['links']['self'], timeout=60) as response, temporary.open('wb') as stream:
        total = 0
        for block in iter(lambda: response.read(4 * 1024 * 1024), b''):
            stream.write(block)
            total += len(block)
            if total // (128 * 1024 * 1024) != (total - len(block)) // (128 * 1024 * 1024):
                print(path.name, round(total / 1e6), 'MB', flush=True)
    assert temporary.stat().st_size == item['size'], path.name
    assert digest(temporary) == expected, path.name
    temporary.replace(path)
    print('Verified', path.name, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Download all 4.1GB and check full archive MD5s')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen('https://zenodo.org/api/records/4904800', timeout=60) as response:
        record = json.load(response)
    (OUT / 'record.json').write_text(json.dumps(record, indent=2))
    if args.full:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(fetch, record['files']))
    else:
        selected_download(record)
