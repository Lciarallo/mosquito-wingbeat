import hashlib
import json
from pathlib import Path
import sys
import pytest
from scripts import restore_publication_artifacts as restore


@pytest.mark.parametrize('value', ['../outside.bin', '/tmp/outside.bin', 'data/../../outside.bin'])
def test_restore_rejects_paths_outside_repository(tmp_path, monkeypatch, value):
    monkeypatch.setattr(restore, 'ROOT', tmp_path)
    with pytest.raises(ValueError):
        restore.target(value)


def test_restore_refuses_to_overwrite_changed_existing_artifact(tmp_path, monkeypatch):
    existing = tmp_path / 'model.bin'
    existing.write_bytes(b'local modified model')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'files': [{'path': 'model.bin', 'sha256': hashlib.sha256(b'original').hexdigest()}]}))
    monkeypatch.setattr(restore, 'ROOT', tmp_path)
    monkeypatch.setattr(restore, 'MANIFEST', manifest)
    monkeypatch.setattr(sys, 'argv', ['restore', '--verify'])
    with pytest.raises(SystemExit, match='no overwrite allowed'):
        restore.main()
    assert existing.read_bytes() == b'local modified model'


def test_restore_verifies_matching_artifacts_without_download(tmp_path, monkeypatch, capsys):
    existing = tmp_path / 'model.bin'
    existing.write_bytes(b'original model')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'files': [{'path': 'model.bin', 'sha256': hashlib.sha256(existing.read_bytes()).hexdigest()}]}))
    monkeypatch.setattr(restore, 'ROOT', tmp_path)
    monkeypatch.setattr(restore, 'MANIFEST', manifest)
    monkeypatch.setattr(sys, 'argv', ['restore', '--verify'])
    restore.main()
    assert json.loads(capsys.readouterr().out) == {'verified': 1, 'missing': 0}
