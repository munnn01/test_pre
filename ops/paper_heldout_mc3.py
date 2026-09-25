#!/usr/bin/env python
"""Evaluate frozen V2 bitstream choices on an unseen mc3_18 analyzer.

The selector reads only the prior two-analyzer, label-free cache. mc3_18 and
ground-truth labels are used *after* selection, solely for evaluation. This
tests model transfer, not a new data holdout: the 1,000 clips are the old TEST.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import cv2
import numpy as np
import torch
import torchvision

from ops.codec_search_ar import QPS, as_video
from ops.dual_codec_search import digest, write_json
from ops.dual_codec_search_confirm_1000 import (CONFIG, file_sha256,
                                                load_frozen, sample_plan)
from ops.merge_dual_codec_search_confirm_1000 import fingerprint, load_shards
from ops.rcts_pilot import clip_id
from src.codecs.standard import StandardCodec, ffmpeg_available
from src.metrics.bd_rate import bd_metric, bd_rate
from src.models.codec_search import make_candidates, normalized_bpp
from src.models.dual_codec_search import select
from src.tasks.action_recognition import ActionRecognitionAnalyzer, kinetics_categories

MODEL = "mc3_18"


def curves(records: list[dict]) -> tuple[dict, dict]:
    """Average rate and Top-1 over videos, preserving QP pairing."""
    if not records:
        raise ValueError("no held-out records")
    curves_by_arm = {}
    for arm in ("anchor", "trial"):
        curve = {}
        for position, qp in enumerate(QPS):
            points = [r["measurements"][position][arm] for r in records]
            if any(r["measurements"][position]["qp"] != qp for r in records):
                raise ValueError("QP order differs from frozen experiment")
            curve[str(qp)] = {"n": len(points),
                              "bpp": float(np.mean([p["bpp"] for p in points])),
                              "top1": float(np.mean([p["correct"] for p in points]))}
        curves_by_arm[arm] = curve
    return curves_by_arm["anchor"], curves_by_arm["trial"]


def compare(curve_a: dict, curve_b: dict) -> dict:
    a, b = [curve_a[str(q)] for q in QPS], [curve_b[str(q)] for q in QPS]
    return {"bd_rate_top1_pct": bd_rate([p["bpp"] for p in a], [p["top1"] for p in a],
                                         [p["bpp"] for p in b], [p["top1"] for p in b]),
            "bd_accuracy_top1_pp": 100 * bd_metric(
                [p["bpp"] for p in a], [p["top1"] for p in a],
                [p["bpp"] for p in b], [p["top1"] for p in b]),
            "min_same_qp_top1_gap_pp": 100 * min(x["top1"] - y["top1"]
                                                for x, y in zip(b, a))}


def summarize(records: list[dict], draws: int) -> dict:
    if draws < 1:
        raise ValueError("bootstrap draws must be positive")
    anchor, trial = curves(records)
    point = compare(anchor, trial)
    rng = np.random.default_rng(20260924)
    samples = {key: [] for key in ("bd_rate_top1_pct", "bd_accuracy_top1_pp")}
    for _ in range(draws):
        picked = [records[i] for i in rng.integers(0, len(records), len(records))]
        a, b = curves(picked)
        metric = compare(a, b)
        for key, values in samples.items():
            if np.isfinite(metric[key]):
                values.append(metric[key])
    intervals = {key: {"valid_draws": len(values), "requested_draws": draws,
                       "ci95": np.percentile(values, [2.5, 97.5]).tolist()
                       if values else None}
                 for key, values in samples.items()}
    timing = {}
    for field in ("encode_decode_s", "inference_s"):
        for arm in ("anchor", "trial"):
            values = [m[arm][field] for row in records for m in row["measurements"]]
            timing[f"{arm}_{field}"] = {"median": float(np.median(values)),
                                         "mean": float(np.mean(values)),
                                         "n": len(values)}
    return {"n": len(records), "model": MODEL, "anchor_curve": anchor,
            "trial_curve": trial, "metrics": point, "bootstrap": intervals,
            "timing": timing, "timing_scope":
            "Only re-encodes identity and selected stream for mc3 evaluation; "
            "not the full six-candidate selector cost."}


def timed_inference(analyzer: ActionRecognitionAnalyzer, video: np.ndarray) -> tuple[int, float]:
    if next(analyzer.parameters()).is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        logits = analyzer.predict(as_video(video).to(next(analyzer.parameters()).device))
    if next(analyzer.parameters()).is_cuda:
        torch.cuda.synchronize()
    return int(logits.argmax(1).item()), time.perf_counter() - start


def evaluate_clip(dataset, index: int, cache: dict, state: dict, policy: dict,
                  analyzer: ActionRecognitionAnalyzer, codec: StandardCodec) -> dict:
    source, label, meta = dataset[index]
    if meta["sequence_id"] != cache["sequence_id"]:
        raise ValueError("cached and decoded video IDs disagree")
    cap = cv2.VideoCapture(dataset.samples[index]["path"])
    ok, _ = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"undecodable source: {meta['sequence_id']}")
    rgb = (source.permute(1, 2, 3, 0).numpy() * 255).round().astype(np.uint8)
    variants = make_candidates(rgb)
    measurements = []
    for item in cache["measurements"]:
        qp = item["qp"]
        picked = select(item["candidates"], qp, policy, state)
        chosen = item["candidates"][picked]
        streams = {}
        for arm, name in (("anchor", "identity128"), ("trial", chosen["name"])):
            if name in streams:
                reused = dict(streams[name])
                reused["encode_decode_s"] = 0.0
                reused["inference_s"] = 0.0
                reused["reused_from_anchor"] = True
                streams[arm] = reused
                continue
            candidate = variants[name]
            _t, h, w, _ = candidate.shape
            start = time.perf_counter()
            reconstructed, native_bpp = codec._encode_decode_clip(candidate, qp=qp)
            encode_decode_s = time.perf_counter() - start
            prediction, inference_s = timed_inference(analyzer, reconstructed)
            value = {"name": name, "bpp": normalized_bpp(native_bpp, h, w),
                     "correct": bool(prediction == label),
                     "encode_decode_s": encode_decode_s,
                     "inference_s": inference_s,
                     "cached_bpp": chosen["bpp"] if arm == "trial"
                     else item["candidates"][0]["bpp"]}
            streams[name] = value
            streams[arm] = value
        measurements.append({"qp": qp, "chosen": chosen["name"],
                             "anchor": streams["anchor"], "trial": streams["trial"]})
    return {"sequence_id": meta["sequence_id"], "measurements": measurements}


def run(args) -> None:
    if not ffmpeg_available():
        raise ValueError("ffmpeg and ffprobe are required")
    if kinetics_categories(MODEL) != kinetics_categories("r3d_18"):
        raise ValueError("mc3_18 Kinetics class order differs from training labels")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows, _ = load_shards(args.cache_dir, args.codec, config)
    by_id = {r["sequence_id"]: r for r in rows}
    state, frozen, policy = load_frozen(args.codec, config)
    dataset, all_indices = sample_plan(args.index, config)
    indices = [i for pos, i in enumerate(all_indices) if pos % 2 == args.shard]
    if len(indices) != 500 or any(clip_id(dataset.samples[i]) not in by_id for i in indices):
        raise ValueError("incomplete or mismatched cached 500-video shard")
    manifest = {"experiment": "dual_v2_heldout_mc3", "codec": args.codec,
                "shard": args.shard, "model": MODEL, "n": len(indices),
                "code_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                "config_sha256": file_sha256(CONFIG),
                "index_sha256": file_sha256(args.index),
                "source_cache": [
                    {"manifest_sha256": file_sha256(directory / "manifest.json"),
                     "records_sha256": file_sha256(directory / "shard_records.jsonl"),
                     "result_sha256": file_sha256(directory / "shard_result.json")}
                    for directory in sorted(args.cache_dir, key=str)],
                "sample_ids": [clip_id(dataset.samples[i]) for i in indices],
                "shard_fingerprint": fingerprint(
                    [clip_id(dataset.samples[i]) for i in indices]),
                "test_fingerprint": config["test_fingerprint"],
                "frozen_policy_sha256": digest(frozen), "risk_sha256": digest(state),
                "policy": policy, "qps": list(QPS),
                "versions": {"python": platform.python_version(),
                             "torch": torch.__version__,
                             "torchvision": torchvision.__version__,
                             "ffmpeg": subprocess.check_output(
                                 ["ffmpeg", "-version"], text=True).splitlines()[0]},
                "hardware": {"logical_cpus": os.cpu_count(),
                             "gpu": torch.cuda.get_device_name(0)
                             if torch.cuda.is_available() else None},
                "limitation": "same previously inspected 1,000-video TEST sample"}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("output cache belongs to a different held-out run")
    write_json(manifest_path, manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analyzer = ActionRecognitionAnalyzer(MODEL, clip_size=112).freeze().to(device)
    codec = StandardCodec(args.codec, preset=config["preset"], strict_decode=True)
    records = []
    for number, index in enumerate(indices, 1):
        path = args.out_dir / "cache" / f"clip_{number:04d}.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
        else:
            key = clip_id(dataset.samples[index])
            record = evaluate_clip(dataset, index, by_id[key], state, policy, analyzer, codec)
            write_json(path, record)
        if record["sequence_id"] != clip_id(dataset.samples[index]):
            raise ValueError("stale per-video mc3 cache")
        records.append(record)
        print(f"[mc3] {args.codec} shard={args.shard} {number}/500", flush=True)
    record_path = args.out_dir / "shard_records.jsonl"
    record_path.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records),
                           encoding="utf-8")
    write_json(args.out_dir / "shard_result.json", {"manifest": manifest,
               "diagnostic_only": summarize(records, args.bootstrap),
               "warning": "Merge both 500-video shards for paper metrics."})


def merge(args) -> None:
    if len(args.shard_dir) != 2:
        raise ValueError("exactly two mc3 shard directories required")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    bundles = []
    for directory in args.shard_dir:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        records = [json.loads(line) for line in (directory / "shard_records.jsonl")
                   .read_text(encoding="utf-8").splitlines()]
        if (manifest["experiment"] != "dual_v2_heldout_mc3"
                or manifest["codec"] != args.codec or len(records) != 500
                or [r["sequence_id"] for r in records] != manifest["sample_ids"]
                or fingerprint(manifest["sample_ids"]) != manifest["shard_fingerprint"]):
            raise ValueError("invalid held-out shard")
        bundles.append((manifest, records))
    bundles.sort(key=lambda bundle: bundle[0]["shard"])
    if [m["shard"] for m, _ in bundles] != [0, 1]:
        raise ValueError("missing or duplicate shard")
    if any(bundles[0][0][key] != bundles[1][0][key]
           for key in ("experiment", "codec", "model", "qps", "test_fingerprint",
                       "code_commit", "config_sha256", "index_sha256",
                       "source_cache", "frozen_policy_sha256", "risk_sha256",
                       "policy")):
        raise ValueError("held-out shard provenance mismatch")
    rows = bundles[0][1] + bundles[1][1]
    ids = [r["sequence_id"] for r in rows]
    if (len(set(ids)) != 1000 or fingerprint(ids) != config["test_fingerprint"]):
        raise ValueError("merged 1,000-video fingerprint mismatch")
    report = {"experiment": "dual_v2_heldout_mc3", "codec": args.codec,
              "split": "previously_inspected_test", "fingerprint": fingerprint(ids),
              "source_manifests": [m for m, _ in bundles],
              "result": summarize(rows, args.bootstrap)}
    write_json(args.out, report)
    print(json.dumps({"codec": args.codec, "n": len(rows),
                      "metrics": report["result"]["metrics"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--index", type=Path, required=True)
    evaluate_parser.add_argument("--codec", choices=("h264", "h265"), required=True)
    evaluate_parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    evaluate_parser.add_argument("--cache-dir", type=Path, action="append", required=True)
    evaluate_parser.add_argument("--out-dir", type=Path, required=True)
    evaluate_parser.add_argument("--bootstrap", type=int, default=200)
    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--codec", choices=("h264", "h265"), required=True)
    merge_parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    merge_parser.add_argument("--out", type=Path, required=True)
    merge_parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    (run if args.command == "evaluate" else merge)(args)


if __name__ == "__main__":
    main()
