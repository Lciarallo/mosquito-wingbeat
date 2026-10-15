"""Acquire the public Torres dataset and checkpoints at a pinned author commit."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from zipfile import ZipFile

import requests

from torres_lstm import AUTHOR_COMMIT, AUTHOR_REPOSITORY, digest, write_json

ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / "data/external/torres_2025"
FILES = ["Dataset/Melspec.zip", "Model/LSTM_Bid_2H_256.pth", "Model/LSTM_Uni_2H_256.pth",
         "Healthy and Infected mosquitos/healthy0001.wav",
         "Healthy and Infected mosquitos/infected0001.wav",
         "README.md", "Training_LSTM.py", "Testing_LSTM_MC_DO.py"]


def fetch(destination=DESTINATION):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    api = "https://api.github.com/repos/NakanoMariko/Dengue-Infected-Mosquito-Detection"
    response = requests.get(api + "/git/trees/" + AUTHOR_COMMIT,
                            params={"recursive": 1}, timeout=90)
    response.raise_for_status()
    tree = response.json()
    if tree.get("truncated"):
        raise ValueError("Author tree response is incomplete.")
    entries = {item["path"]: item for item in tree["tree"]}
    manifest = dict(repository=AUTHOR_REPOSITORY, commit=AUTHOR_COMMIT, files=[])
    for name in FILES:
        item = entries[name]
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://raw.githubusercontent.com/NakanoMariko/Dengue-Infected-Mosquito-Detection/" + AUTHOR_COMMIT + "/" + quote(name, safe="/")
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".download")
            with requests.get(url, timeout=120, stream=True) as stream:
                stream.raise_for_status()
                with temporary.open("wb") as target:
                    for block in stream.iter_content(1024 * 1024):
                        target.write(block)
            temporary.replace(path)
        value = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(value)).encode() + b"\0" + value).hexdigest()
        if len(value) != item["size"] or blob != item["sha"]:
            raise ValueError("Pinned source digest mismatch: " + name)
        manifest["files"].append(dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                                      source_path=name, url=url, size=len(value),
                                      sha256=digest(path), git_blob_sha1=blob))
        print(json.dumps(dict(downloaded=name, verified=True, bytes=len(value))), flush=True)
    with ZipFile(destination / "Dataset/Melspec.zip") as archive:
        members = archive.infolist()
        if len(members) != 1 or members[0].filename != "Melspec.json" or members[0].file_size > 100_000_000:
            raise ValueError("Unexpected dataset archive schema.")
        target = destination / "Dataset/Melspec.json"
        data = archive.read(members[0])
        if not target.exists() or target.read_bytes() != data:
            target.write_bytes(data)
        manifest["expanded_json_sha256"] = digest(target)
    write_json(destination / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    args = parser.parse_args()
    fetch(args.destination)
