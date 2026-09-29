"""Download the complete Dryad dataset and verify publisher MD5 checksums."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import zipfile

import requests

BASE_URL = "https://datadryad.org"
DATASET_URL = BASE_URL + "/api/v2/datasets/doi%3A10.5061%2Fdryad.98d7s"
ROOT = Path(__file__).resolve().parent


def hashes(path: Path) -> dict:
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(block)
            sha256.update(block)
    return {"md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def get_json(url: str) -> dict:
    for attempt in range(4):
        response = requests.get(url, timeout=(20, 120))
        if response.status_code == 429 or response.status_code >= 500:
            if attempt < 3:
                delay = response.headers.get("Retry-After", "")
                time.sleep(min(int(delay), 30) if delay.isdigit() else 2 ** attempt)
                continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError("Unreachable")


def metadata(data: Path) -> list[dict]:
    cache_names = ["dryad_dataset.json", "dryad_version.json", "dryad_files.json", "dryad_files_all.json"]
    if all((data / name).exists() for name in cache_names):
        dataset, version, first_page, files = [json.loads((data / name).read_text()) for name in cache_names]
        pages = [first_page]
    else:
        dataset = get_json(DATASET_URL)
        version = get_json(BASE_URL + dataset["_links"]["stash:version"]["href"])
        url = BASE_URL + version["_links"]["stash:files"]["href"]
        files, pages = [], []
        while url:
            page = get_json(url)
            pages.append(page)
            files.extend(page["_embedded"]["stash:files"])
            next_link = page["_links"].get("next")
            url = BASE_URL + next_link["href"] if next_link else None
    assert len(files) == pages[0]["total"], "Incomplete API pagination"
    # API downloads require authentication; the published landing page exposes
    # anonymous download links. Discover those links rather than inventing URLs.
    if not all("public_download_url" in info for info in files):
        landing = requests.get(BASE_URL + "/dataset/doi:10.5061/dryad.98d7s", timeout=60)
        landing.raise_for_status()
        public_links = {href.rsplit("/", 1)[-1]: BASE_URL + href
                        for href in re.findall(r'href="(/downloads/file_stream/\d+)"', landing.text)}
        for info in files:
            file_id = info["_links"]["self"]["href"].rsplit("/", 1)[-1]
            info["public_download_url"] = public_links[file_id]
    mirror_file = data / "zenodo_record.json"
    mirror = json.loads(mirror_file.read_text()) if mirror_file.exists() else get_json("https://zenodo.org/api/records/4964774")
    mirror_file.write_text(json.dumps(mirror, indent=2))
    assert mirror["metadata"]["doi"] == "10.5061/dryad.98d7s"
    mirrored = {entry["key"]: entry for entry in mirror["files"]}
    assert set(mirrored) == {entry["path"] for entry in files}
    for info in files:
        copy = mirrored[info["path"]]
        assert copy["size"] == info["size"] and copy["checksum"] == "md5:" + info["digest"]
        info["zenodo_download_url"] = copy["links"]["self"]
    for name, obj in [("dryad_dataset.json", dataset), ("dryad_version.json", version),
                      ("dryad_files.json", pages[0]), ("dryad_files_all.json", files)]:
        (data / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    return files


def download_file(info: dict, archive_dir: Path) -> dict:
    filename = Path(info["path"]).name
    target = archive_dir / filename
    # The Dryad API returns 401 and the public download routes return 403 in
    # this environment. Its Zenodo mirror has the same DOI and verified hashes.
    url = info["zenodo_download_url"]
    for attempt in range(4):
        try:
            if target.exists() and target.stat().st_size == info["size"]:
                digest = hashes(target)
                if digest["md5"] == info["digest"]:
                    return {"filename": filename, "size": target.stat().st_size,
                            "url": url, "official_md5": info["digest"], **digest}
            temporary = target.with_suffix(target.suffix + ".part")
            with requests.get(url, stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            stream.write(block)
            digest = hashes(temporary)
            if temporary.stat().st_size != info["size"] or digest["md5"] != info["digest"]:
                raise ValueError(f"Size or MD5 mismatch: {filename}")
            temporary.replace(target)
            print(f"Verified: {filename} ({info['size'] / 1e6:.1f} MB)", flush=True)
            return {"filename": filename, "size": target.stat().st_size,
                    "url": url, "official_md5": info["digest"], **digest}
        except Exception as error:
            if attempt == 3:
                raise
            print(f"Retry {attempt + 1}: {filename}: {error}", flush=True)
            time.sleep(2 ** attempt)
    raise RuntimeError("Unreachable")


def extract_file(info: dict, data: Path) -> None:
    import rarfile
    import libarchive

    archive = data / "archives" / Path(info["path"]).name
    destination = data / "raw" / archive.stem
    marker = destination / ".extracted.json"
    if marker.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    with rarfile.RarFile(archive) as handle:
        for member in handle.infolist():
            name = member.filename.replace("\\", "/")
            resolved = (destination / name).resolve()
            if not resolved.is_relative_to(destination.resolve()):
                raise ValueError(f"Unsafe archive member: {name}")
    with libarchive.file_reader(str(archive)) as entries:
        for member in entries:
            target = destination / member.pathname.replace("\\", "/")
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f"Unsafe archive member: {member.pathname}")
            if member.isdir:
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as stream:
                    for block in member.get_blocks():
                        stream.write(block)
            else:
                raise ValueError(f"Unsupported archive entry: {member.pathname}")
    marker.write_text(json.dumps({"archive": archive.name, "md5": info["digest"]}))
    print(f"Extracted: {archive.name}", flush=True)


def fetch_dataset(root: Path = ROOT, workers: int = 3, extract: bool = True) -> list[dict]:
    data = root / "data"
    archive_dir = data / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    files = metadata(data)
    print(f"Dryad: {len(files)} files, {sum(f['size'] for f in files) / 1e9:.3f} GB", flush=True)
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_file, info, archive_dir): info for info in files}
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda item: item["filename"])
    manifest = {"dataset_doi": "10.5061/dryad.98d7s", "retrieved_utc":
                datetime.now(timezone.utc).isoformat(), "files": records,
                "total_bytes": sum(r["size"] for r in records)}
    (data / "download_manifest.json").write_text(json.dumps(manifest, indent=2))
    if extract:
        for info in files:
            extract_file(info, data)
    return records


def fetch_noise(root: Path = ROOT) -> dict:
    """Fetch the authors' real background noises at a pinned Git commit."""
    commit = "27fdf026728c66c3a8448454987aa18b904cfce2"
    url = f"https://raw.githubusercontent.com/Mjrovai/wingbeat-mosquito-tinyml/{commit}/dataset/noise.zip"
    expected = "5f4ba031c47b4766d09d13b71a7991b8d153ba3e635c623be480128e7d3895e5"
    archive = root / "data/archives/tinyml_noise.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists() or hashes(archive)["sha256"] != expected:
        response = requests.get(url, timeout=(20, 120))
        response.raise_for_status()
        temporary = archive.with_suffix(".zip.part")
        temporary.write_bytes(response.content)
        assert hashes(temporary)["sha256"] == expected, "Noise archive checksum mismatch"
        temporary.replace(archive)
    destination = root / "data/noise"
    if not (destination / ".extracted").exists():
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                if not (destination / member.filename).resolve().is_relative_to(destination.resolve()):
                    raise ValueError(f"Unsafe ZIP entry: {member.filename}")
            handle.extractall(destination)
        (destination / ".extracted").touch()
    result = {"source_url": url, "commit": commit, "size": archive.stat().st_size,
              "sha256": expected, "label": "noise as provided by Altayeb et al. authors"}
    (root / "data/noise_manifest.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()
    fetch_dataset(workers=args.workers, extract=not args.no_extract)
