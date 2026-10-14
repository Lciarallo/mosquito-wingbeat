"""Download the authors' exact feature archive, using verified byte ranges."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "data/external/faithful_2026_10_03/dr_biol"
URL = "https://zenodo.org/api/records/17287037"


def digest(path, algorithm="md5"):
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as r:
        record = json.load(r)
    (DEST / "record.json").write_text(json.dumps(record, indent=2))
    entry = record["files"][0]
    assert entry["key"] == "Mosquito_sound_8species_4domains.pickle.zip"
    archive = DEST / entry["key"]
    expected = entry["checksum"].split(":", 1)[1]
    if not archive.exists() or digest(archive) != expected:
        size = entry["size"]
        chunk_size = 64 * 1024 * 1024
        chunks = [(i, start, min(size - 1, start + chunk_size - 1))
                  for i, start in enumerate(range(0, size, chunk_size))]
        part_dir = DEST / "ranges"
        part_dir.mkdir(exist_ok=True)

        def fetch(chunk):
            i, start, end = chunk
            target = part_dir / f"{i:03d}.part"
            if target.exists() and target.stat().st_size == end - start + 1:
                return i
            for attempt in range(4):
                try:
                    req = urllib.request.Request(entry["links"]["self"],
                                                 headers={"Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(req, timeout=90) as r:
                        assert r.status == 206
                        assert r.getheader("Content-Range") == f"bytes {start}-{end}/{size}"
                        with target.open("wb") as f:
                            for block in iter(lambda: r.read(1024 * 1024), b""):
                                f.write(block)
                    assert target.stat().st_size == end - start + 1
                    return i
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(fetch, c) for c in chunks]
            for n, future in enumerate(as_completed(futures), 1):
                print(f"verified range {future.result()}: {n}/{len(chunks)}", flush=True)
        assembling = archive.with_suffix(".assembling")
        with assembling.open("wb") as f:
            for i, _, _ in chunks:
                with (part_dir / f"{i:03d}.part").open("rb") as p:
                    for block in iter(lambda: p.read(8 * 1024 * 1024), b""):
                        f.write(block)
        assert assembling.stat().st_size == size
        assert digest(assembling) == expected, "Archive differs from the authors' checksum"
        assembling.replace(archive)
    assert digest(archive) == expected
    with zipfile.ZipFile(archive) as z:
        members = [n for n in z.namelist()
                   if Path(n).name == "Mosquito_sound_8species_4domains.pickle"]
        assert len(members) == 1
        dest = DEST / "Mosquito_sound_8species_4domains.pickle"
        if not dest.exists() or dest.stat().st_size != z.getinfo(members[0]).file_size:
            with z.open(members[0]) as p, dest.open("wb") as f:
                for block in iter(lambda: p.read(8 * 1024 * 1024), b""):
                    f.write(block)
    print("author archive verified:", expected, "pickle:", dest.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
