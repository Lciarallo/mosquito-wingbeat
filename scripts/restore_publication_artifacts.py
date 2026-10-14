"""Restore original research artifacts from the GitHub snapshot without overwriting changes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'reports/github_publication_2026_10_14/artifact_manifest.json'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def target(path: str) -> Path:
    result = (ROOT / path).resolve()
    if not result.is_relative_to(ROOT) or Path(path).is_absolute():
        raise ValueError(f'Unsafe artifact path: {path}')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='Verify local artifacts without downloading.')
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    missing = []
    for item in manifest['files']:
        path = target(item['path'])
        if path.exists():
            if not path.is_file() or sha256(path) != item['sha256']:
                raise SystemExit(f'Existing artifact differs; no overwrite allowed: {item["path"]}')
        else:
            missing.append(item)
    if args.verify:
        print(json.dumps({'verified': len(manifest['files']) - len(missing), 'missing': len(missing)}))
        if missing:
            raise SystemExit(1)
        return
    assets_needed = {item['asset'] for item in missing}
    with tempfile.TemporaryDirectory(prefix='mosquito-artifacts-') as cache:
        for asset in manifest['assets']:
            if asset['name'] not in assets_needed:
                continue
            subprocess.run(['gh', 'release', 'download', manifest['tag'], '--repo', manifest['repository'],
                            '--pattern', asset['name'], '--dir', cache], check=True)
            archive = Path(cache) / asset['name']
            if sha256(archive) != asset['sha256']:
                raise SystemExit(f'Archive hash mismatch: {asset["name"]}')
            allowed = {item['path']: item for item in manifest['files'] if item['asset'] == asset['name']}
            found = set()
            with tarfile.open(archive, 'r|') as bundle:
                for member in bundle:
                    if member.name not in allowed or not member.isfile() or member.name in found:
                        raise SystemExit(f'Unexpected archive entry: {member.name}')
                    found.add(member.name)
                    item = allowed[member.name]
                    path = target(member.name)
                    if member.size != item['size']:
                        raise SystemExit(f'Artifact size mismatch: {member.name}')
                    if path.exists():
                        if sha256(path) != item['sha256']:
                            raise SystemExit(f'Existing artifact changed: {member.name}')
                        continue
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_name(path.name + '.restore-partial')
                    try:
                        with bundle.extractfile(member) as source, temporary.open('xb') as dest:
                            for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
                                dest.write(chunk)
                        if sha256(temporary) != item['sha256']:
                            raise SystemExit(f'Artifact hash mismatch: {member.name}')
                        if path.exists():
                            raise SystemExit(f'Artifact appeared during restoration: {member.name}')
                        temporary.rename(path)
                    finally:
                        if temporary.exists():
                            temporary.unlink()
            if found != set(allowed):
                raise SystemExit(f'Archive entries missing: {asset["name"]}')
            archive.unlink()
    # Keep restored large binaries out of regular commits without changing frozen .gitignore.
    gitdir = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', '--git-path', 'info/exclude'], text=True).strip()
    exclude = Path(gitdir)
    if not exclude.is_absolute():
        exclude = ROOT / exclude
    content = exclude.read_text() if exclude.exists() else ''
    patterns = ['/' + item['path'] for item in manifest['files']]
    new = [pattern for pattern in patterns if pattern not in content.splitlines()]
    if new:
        exclude.parent.mkdir(parents=True, exist_ok=True)
        with exclude.open('a') as stream:
            stream.write('\n# Research snapshot assets restored from GitHub\n' + '\n'.join(new) + '\n')
    print(json.dumps({'restored': len(missing), 'verified': len(manifest['files'])}))


if __name__ == '__main__':
    main()
