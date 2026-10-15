"""Post-freeze, metadata-only acquisition of unused positive clips.

This is an additional corpus transfer check, not a new biological campaign.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import struct
import subprocess
import zlib

import pandas as pd
import soundfile as sf

from ternary_protocol import ROOT, now, digest, json_write, read_audio, pcm_signatures, verify_freeze
from ternary_reporting import verify_evaluation
from predict_ternary import predict_audio
from fetch_humbug import byte_range

INDEX = ROOT / "tmp/reproductions_2026_10_01/cdmsc_central_full.json"
OFFICIAL_SOURCE = "https://zenodo.org/records/19095788"


def index_audio(path):
    try:
        return digest(path), pcm_signatures(read_audio(path, 8000)), None
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        return digest(path), set(), {"path": str(path), "error": type(error).__name__}


def extract_entry(raw, entry):
    header = struct.unpack_from("<4s5H3I2H", raw)
    if header[0] != b"PK\x03\x04":
        raise ValueError("Invalid ZIP local header.")
    start = 30 + header[-2] + header[-1]
    data = raw[start:start + entry["compressed"]]
    if entry["method"] == 8:
        data = zlib.decompress(data, -15)
    elif entry["method"] != 0:
        raise ValueError("Unsupported ZIP compression.")
    if len(data) != entry["uncompressed"] or zlib.crc32(data) != entry["crc"]:
        raise ValueError("ZIP size/CRC mismatch.")
    return data


def acquire(run, search_receipt):
    run = Path(run).resolve()
    freeze = verify_freeze(run)
    verify_evaluation(run)
    receipt = json.loads(Path(search_receipt).read_text())
    if (receipt.get("official_source") != OFFICIAL_SOURCE
            or receipt.get("freeze_sha256") != digest(run / "freeze.json")
            or receipt.get("checked_utc", "") <= freeze["created_utc"]
            or not receipt.get("official_page_read")):
        raise ValueError("A real official-source lookup after this freeze is required.")
    protocol = json.loads((run / "protocol.json").read_text())
    if digest(INDEX) != protocol["postfreeze_acquisition"]["archive_index_sha256"]:
        raise ValueError("Archive metadata changed after protocol declaration.")
    central = json.loads(INDEX.read_text())
    if central.get("partial_central_directory"):
        raise ValueError("Incomplete ZIP index.")
    out = run / "postfreeze"
    out.mkdir(exist_ok=False)
    (out / "audio").mkdir()
    json_write(out / "search_receipt.json", receipt)
    known = set()
    for folder in ["data/raw", "data/external", "data/noise", "data/external_datasets",
                   "tmp", "external", "results"]:
        known.update(p.resolve() for p in (ROOT / folder).rglob("*.wav") if out not in p.parents)
    sources = pd.read_csv(run / "source_pcm_audit.csv")
    known.update((ROOT / name).resolve() for name in sources.path)
    known = sorted(known)
    names = {p.name for p in known}
    candidates = sorted(
        [e for e in central["entries"] if Path(e["name"]).name.startswith("S_1_D_3_")
         and e["name"].endswith(".wav") and Path(e["name"]).name not in names],
        key=lambda e: int(Path(e["name"]).stem.split("_")[-1]),
    )[:30]
    selection = {
        "created_utc": now(), "run_id": freeze["run_id"], "frozen_utc": freeze["created_utc"],
        "freeze_sha256": digest(run / "freeze.json"), "official_source": OFFICIAL_SOURCE,
        "archive_index_sha256": digest(INDEX), "candidates": candidates, "selected": [],
        "accepted": [], "excluded": [], "known_files": len(known),
        "rule": protocol["postfreeze_acquisition"]["rule"],
        "familiar_corpus": True, "new_biological_campaign": False,
        "independent_individuals_sessions_established": False,
    }
    json_write(out / "selection.json", selection)
    known_bytes, known_pcm, errors = set(), set(), []
    with ThreadPoolExecutor(max_workers=4) as workers:
        for i, (bytehash, keys, error) in enumerate(workers.map(index_audio, known), 1):
            known_bytes.add(bytehash)
            known_pcm.update(keys)
            if error:
                errors.append(error)
            if i % 1000 == 0 or i == len(known):
                print("POST-FREEZE NOVELTY", i, "/", len(known), flush=True)
    json_write(out / "novelty_audit.json", {
        "known_files": len(known), "byte_hashes": len(known_bytes), "pcm_keys": len(known_pcm),
        "decode_errors": errors, "complete_decoding_coverage": not errors,
        "limits": "Whole/aligned 1/.992/.3 s 8kHz raw/gain-normalised nonconstant PCM. "
                  "Arbitrary transformations and biological/session ancestry are not established.",
    })
    for entry in candidates:
        raw = byte_range(central["file"]["links"]["self"], entry["offset"],
                         entry["offset"] + 30 + len(entry["name"]) + entry["compressed"] + 1024)
        data = extract_entry(raw, entry)
        path = out / "audio" / Path(entry["name"]).name
        path.write_bytes(data)
        seconds = sf.info(path).duration
        item = {"path": str(path.relative_to(ROOT)), "reference_class": 2,
                "reference_species": "Aedes aegypti", "seconds": seconds,
                "sha256": digest(path), "zip_crc32": entry["crc"]}
        if seconds < 1.984:
            selection["excluded"].append(dict(item, reason="duration_below_predeclared_horizon"))
            continue
        selection["selected"].append(item)
        if len(selection["selected"]) == 2:
            break
    # Novelty exclusion never triggers replacement based on a model's outcome.
    for item in selection["selected"]:
        keys = pcm_signatures(read_audio(ROOT / item["path"], 8000))
        overlap = len(keys & known_pcm)
        duplicate = item["sha256"] in known_bytes
        item = dict(item, known_pcm_matches=overlap, byte_duplicate=duplicate)
        if overlap or duplicate:
            selection["excluded"].append(dict(item, reason="known_byte_or_declared_pcm_collision"))
        else:
            selection["accepted"].append(item)
            known_bytes.add(item["sha256"])
            known_pcm.update(keys)
    selection["selection_complete_utc"] = now()
    json_write(out / "selection_complete.json", selection)
    results = []
    for item in selection["accepted"]:
        summary = predict_audio(run, ROOT / item["path"], out / (Path(item["path"]).stem + "_inference"))
        results.append(dict(item, predicted_class=summary["predicted_class"], decision=summary["decision"],
                            correct=summary["predicted_class"] == 2, policies=summary["policies"],
                            quality_invalid_windows=summary["quality_invalid_windows"]))
    json_write(out / "results.json", {
        "completed_utc": now(), "run_id": freeze["run_id"], "records": results,
        "correct_aedes": sum(item["correct"] for item in results), "total": len(results),
        "only_positive_audio": True, "precision_specificity_not_estimable": True,
        "parameters_retuned": False, "new_biological_campaign": False,
    })
    verify_freeze(run)
    print("POST-FREEZE COMPLETE", sum(item["correct"] for item in results), "/", len(results), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--search-receipt", type=Path, required=True)
    args = parser.parse_args()
    acquire(args.run, args.search_receipt)


if __name__ == "__main__":
    main()
