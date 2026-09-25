#!/usr/bin/env python
"""Collect mc3_18 candidate outcomes on the V2 TRAIN fit/calibration videos.

Only TRAIN-fit and disjoint TRAIN-calibration are measured. This is a V4
development dataset, not an independent evaluation or a new source holdout.
The frozen r2plus1d_18/r3d_18 candidate cache supplies label-free selector
signals; mc3 outcomes collected here become TRAIN labels only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

from ops.dual_codec_lowqp_v3 import load_pilot
from ops.dual_codec_search import digest, write_json
from ops.paper_heldout_mc3 import timed_inference
from ops.rcts_pilot import clip_id
from src.codecs.standard import StandardCodec, ffmpeg_available
from src.data.video_dataset import VideoClipDataset
from src.models.codec_search import CANDIDATES, make_candidates, normalized_bpp
from src.tasks.action_recognition import ActionRecognitionAnalyzer, kinetics_categories

REPO = Path(__file__).resolve().parents[1]
QPS = (30, 35, 40)
STAGE_COUNTS = {"fit": 400, "calibration": 200}
MODEL = "mc3_18"


def source_motion_features(rgb: np.ndarray) -> dict[str, float]:
    """Cheap source-only temporal/edge statistics; no task labels or mc3 output."""
    if rgb.ndim != 4 or rgb.shape[0] < 2 or rgb.shape[-1] != 3:
        raise ValueError("expected T,H,W,3 RGB clip")
    gray = cv2.cvtColor(rgb[0], cv2.COLOR_RGB2GRAY)
    temporal = np.abs(np.diff(rgb.astype(np.float32), axis=0)) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return {
        "temporal_mean_abs": float(temporal.mean()),
        "temporal_p90_abs": float(np.percentile(temporal, 90)),
        "temporal_active_fraction": float(np.mean(temporal > (12 / 255))),
        "first_frame_edge_mean": float(np.mean(np.sqrt(gx * gx + gy * gy)) / 1020.0),
    }


def evaluate_clip(dataset: VideoClipDataset, index: int, pilot: dict,
                  analyzer: ActionRecognitionAnalyzer, codec: StandardCodec) -> dict:
    source, label, meta = dataset[index]
    if meta["sequence_id"] != pilot["sequence_id"]:
        raise ValueError("pilot and decoded video IDs differ")
    cap = cv2.VideoCapture(dataset.samples[index]["path"])
    ok, _ = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"undecodable source: {meta['sequence_id']}")
    rgb = (source.permute(1, 2, 3, 0).numpy() * 255).round().astype(np.uint8)
    variants = make_candidates(rgb)
    measurements = []
    for item in pilot["measurements"]:
        qp = item["qp"]
        if qp not in QPS:
            continue
        candidates = []
        for cached in item["candidates"]:
            name = cached["name"]
            candidate = variants[name]
            _t, h, w, _ = candidate.shape
            start = time.perf_counter()
            reconstructed, native_bpp = codec._encode_decode_clip(candidate, qp=qp)
            encode_decode_s = time.perf_counter() - start
            prediction, inference_s = timed_inference(analyzer, reconstructed)
            bpp = normalized_bpp(native_bpp, h, w)
            if abs(bpp - cached["bpp"]) > 1e-9:
                raise ValueError(f"bitstream differs from V2 pilot: {name} QP{qp}")
            candidates.append({"name": name, "bpp": bpp,
                "mc3_correct": bool(prediction == label),
                "encode_decode_s": encode_decode_s, "inference_s": inference_s})
        if [c["name"] for c in candidates] != list(CANDIDATES):
            raise ValueError("candidate order changed")
        measurements.append({"qp": qp, "candidates": candidates})
    if [item["qp"] for item in measurements] != list(QPS):
        raise ValueError("missing critical-QP measurements")
    return {"schema": 1, "sequence_id": meta["sequence_id"],
            "source_features": source_motion_features(rgb),
            "measurements": measurements}


def evaluate(args: argparse.Namespace) -> None:
    if not ffmpeg_available():
        raise ValueError("ffmpeg/ffprobe required")
    if kinetics_categories(MODEL) != kinetics_categories("r3d_18"):
        raise ValueError("mc3 and original V2 class order differ")
    pilot_manifest, _risk, _frozen, pilot = load_pilot(
        args.archive, args.codec, stages=("fit", "calibration"))
    index = json.loads(args.index.read_text(encoding="utf-8"))
    dataset = VideoClipDataset(args.index, split="train", num_frames=16,
                               frame_size=128, temporal_stride=2,
                               train=False, return_metadata=True)
    locations = {clip_id(row): i for i, row in enumerate(dataset.samples)}
    if len(locations) != len(dataset.samples):
        raise ValueError("TRAIN clip IDs are not unique")
    sample_ids = {}
    for stage, expected in STAGE_COUNTS.items():
        ids = pilot_manifest["split_ids"][stage]
        if len(ids) != expected or any(key not in locations for key in ids):
            raise ValueError(f"incomplete {stage} source list")
        sample_ids[stage] = ids[args.shard::2]
        if len(sample_ids[stage]) != expected // 2:
            raise ValueError(f"incorrect {stage} shard length")
    if set(sample_ids["fit"]) & set(sample_ids["calibration"]):
        raise ValueError("fit/calibration clip overlap")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                     cwd=REPO, text=True).strip()
    manifest = {"experiment": "dual_v4_mc3_fit_cache", "codec": args.codec,
        "shard": args.shard, "shards": 2, "qps": list(QPS),
        "candidates": list(CANDIDATES), "model": MODEL,
        "stage_sample_ids": sample_ids,
        "stage_shard_fingerprints": {stage: digest(ids) for stage, ids in sample_ids.items()},
        "stage_full_fingerprints": {stage: digest(pilot_manifest["split_ids"][stage])
                                    for stage in STAGE_COUNTS},
        "pilot_manifest_sha256": digest(pilot_manifest),
        "pilot_archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "index_sha256": hashlib.sha256(args.index.read_bytes()).hexdigest(),
        "code_commit": commit,
        "versions": {"python": platform.python_version(),
                     "torch": torch.__version__, "torchvision": torchvision.__version__},
        "scope": "TRAIN-fit and disjoint TRAIN-calibration only; mc3 is a V4 training target."}
    prior_path = args.out_dir / "manifest.json"
    if prior_path.exists() and json.loads(prior_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("output cache belongs to a different experiment")
    write_json(prior_path, manifest)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analyzer = ActionRecognitionAnalyzer(MODEL, clip_size=112).freeze().to(device)
    codec = StandardCodec(args.codec, preset="medium", strict_decode=True)
    for stage in STAGE_COUNTS:
        rows = []
        for number, position in enumerate(range(args.shard, STAGE_COUNTS[stage], 2), 1):
            key = pilot_manifest["split_ids"][stage][position]
            cache_path = args.out_dir / "cache" / stage / f"clip_{number:04d}.json"
            if cache_path.exists():
                row = json.loads(cache_path.read_text(encoding="utf-8"))
            else:
                row = evaluate_clip(dataset, locations[key], pilot[stage][position],
                                    analyzer, codec)
                write_json(cache_path, row)
            if (row.get("schema") != 1 or row.get("sequence_id") != key
                    or [m["qp"] for m in row["measurements"]] != list(QPS)
                    or any([c["name"] for c in m["candidates"]] != list(CANDIDATES)
                           for m in row["measurements"])):
                raise ValueError("stale or incomplete per-video mc3 cache")
            rows.append(row)
            print(f"[v4-mc3] {args.codec} shard={args.shard} {stage} "
                  f"{number}/{STAGE_COUNTS[stage] // 2}", flush=True)
        path = args.out_dir / f"{stage}_records.jsonl"
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        temporary.replace(path)
    write_json(args.out_dir / "shard_result.json", {"manifest": manifest,
        "counts": {stage: len(sample_ids[stage]) for stage in STAGE_COUNTS},
        "warning": "Training cache only; no V4 test result."})


def merge(args: argparse.Namespace) -> None:
    if len(args.shard_dir) != 2:
        raise ValueError("two shard directories required")
    bundles = []
    for directory in args.shard_dir:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("experiment") != "dual_v4_mc3_fit_cache"
                or manifest.get("codec") != args.codec
                or manifest.get("qps") != list(QPS)
                or manifest.get("candidates") != list(CANDIDATES)):
            raise ValueError("invalid V4 fit-cache shard manifest")
        records = {}
        for stage, count in STAGE_COUNTS.items():
            rows = [json.loads(line) for line in
                    (directory / f"{stage}_records.jsonl").read_text(encoding="utf-8").splitlines()]
            if (len(rows) != count // 2
                    or [row["sequence_id"] for row in rows] != manifest["stage_sample_ids"][stage]
                    or digest(manifest["stage_sample_ids"][stage])
                       != manifest["stage_shard_fingerprints"][stage]):
                raise ValueError(f"invalid {stage} V4 shard")
            records[stage] = rows
        bundles.append((manifest, records))
    bundles.sort(key=lambda pair: pair[0]["shard"])
    if [item[0]["shard"] for item in bundles] != [0, 1]:
        raise ValueError("missing or duplicate V4 shard")
    # Raw index hashes can differ across Kaggle mounts because records contain
    # absolute paths and filesystem enumeration order. The paired pilot IDs,
    # archive hash, and exact per-candidate bitrates are the mount-independent
    # invariants. Preserve each raw index hash in its source manifest instead.
    common = ("experiment", "codec", "qps", "candidates", "model",
              "stage_full_fingerprints", "pilot_manifest_sha256",
              "pilot_archive_sha256", "code_commit")
    if any(bundles[0][0][key] != bundles[1][0][key] for key in common):
        raise ValueError("V4 shard provenance mismatch")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_ids = set()
    for stage, count in STAGE_COUNTS.items():
        ids = [None] * count
        by_id = {}
        for manifest, records in bundles:
            ids[manifest["shard"]::2] = manifest["stage_sample_ids"][stage]
            by_id.update({row["sequence_id"]: row for row in records[stage]})
        if (len(by_id) != count or digest(ids) != bundles[0][0]["stage_full_fingerprints"][stage]
                or all_ids.intersection(ids)):
            raise ValueError(f"merged {stage} IDs overlap or mismatch")
        all_ids.update(ids)
        path = args.out_dir / f"{stage}_records.jsonl"
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for key in ids:
                stream.write(json.dumps(by_id[key], allow_nan=False) + "\n")
        temporary.replace(path)
    write_json(args.out_dir / "manifest.json", {"experiment": "dual_v4_mc3_fit_cache",
        "codec": args.codec, "stage_counts": STAGE_COUNTS,
        "source_manifests": [m for m, _ in bundles],
        "scope": "TRAIN development labels only; no independent test."})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluator = sub.add_parser("evaluate")
    evaluator.add_argument("--archive", type=Path, required=True)
    evaluator.add_argument("--index", type=Path, required=True)
    evaluator.add_argument("--codec", choices=("h264", "h265"), required=True)
    evaluator.add_argument("--shard", type=int, choices=(0, 1), required=True)
    evaluator.add_argument("--out-dir", type=Path, required=True)
    merger = sub.add_parser("merge")
    merger.add_argument("--codec", choices=("h264", "h265"), required=True)
    merger.add_argument("--shard-dir", type=Path, action="append", required=True)
    merger.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    (evaluate if args.command == "evaluate" else merge)(args)


if __name__ == "__main__":
    main()
