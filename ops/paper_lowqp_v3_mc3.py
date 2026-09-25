#!/usr/bin/env python
"""Evaluate V3 low-QP ablations with mc3_18 on the old V2 VAL pilot videos.

The old mc3 TEST result motivated this development experiment.  It is no
longer an unseen-analyzer confirmation, and VAL is not a new source holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]

import cv2
import numpy as np
import torch
import torchvision

from ops.codec_search_ar import QPS, as_video
from ops.dual_codec_lowqp_v3 import ARMS, load_pilot, select_lowqp
from ops.dual_codec_search import digest, write_json
from ops.paper_heldout_mc3 import compare, timed_inference
from ops.rcts_pilot import clip_id
from src.codecs.standard import StandardCodec, ffmpeg_available
from src.data.video_dataset import VideoClipDataset
from src.models.codec_search import make_candidates, normalized_bpp
from src.models.dual_codec_search import observations, risk_scores
from src.tasks.action_recognition import ActionRecognitionAnalyzer, kinetics_categories


def _curves(rows: list[dict], arm: str) -> tuple[dict, dict]:
    curves = {}
    for name in ("identity", arm):
        curves[name] = {}
        for position, qp in enumerate(QPS):
            points = [row["measurements"][position]["streams"][
                "identity128" if name == "identity" else row["measurements"][position]["choices"][arm]
            ] for row in rows]
            curves[name][str(qp)] = {"n": len(points),
                "bpp": float(np.mean([p["bpp"] for p in points])),
                "top1": float(np.mean([p["correct"] for p in points]))}
    return curves["identity"], curves[arm]


def summarize(rows: list[dict], draws: int) -> dict:
    if not rows or draws < 1:
        raise ValueError("nonempty records and positive bootstrap required")
    result = {}
    rng = np.random.default_rng(20260925)
    samples = rng.integers(0, len(rows), size=(draws, len(rows)))
    for arm in ARMS:
        anchor, trial = _curves(rows, arm)
        point = compare(anchor, trial)
        distributions = {key: [] for key in ("bd_rate_top1_pct", "bd_accuracy_top1_pp")}
        for picked in samples:
            a, b = _curves([rows[i] for i in picked], arm)
            metric = compare(a, b)
            for key in distributions:
                if metric[key] is not None and np.isfinite(metric[key]):
                    distributions[key].append(metric[key])
        result[arm] = {"anchor_curve": anchor, "trial_curve": trial,
            "metrics": point,
            "bootstrap": {key: {"valid_draws": len(v), "requested_draws": draws,
                "ci95": np.percentile(v, [2.5, 97.5]).tolist() if v else None}
                for key, v in distributions.items()}}
    return {"n": len(rows), "analyzer": "mc3_18", "arms": result,
        "bootstrap_unit": "whole video, all QPs paired",
        "interpretation": "Exploratory VAL transfer test after examining mc3 on old TEST."}


def evaluate_clip(dataset: VideoClipDataset, index: int, cache: dict,
                  risk: dict, policy: dict, analyzer: ActionRecognitionAnalyzer,
                  codec: StandardCodec) -> dict:
    source, label, meta = dataset[index]
    if meta["sequence_id"] != cache["sequence_id"]:
        raise ValueError("decoded and cached IDs differ")
    cap = cv2.VideoCapture(dataset.samples[index]["path"])
    ok, _ = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"undecodable video: {meta['sequence_id']}")
    rgb = (source.permute(1, 2, 3, 0).numpy() * 255).round().astype(np.uint8)
    variants = make_candidates(rgb)
    result = []
    for item in cache["measurements"]:
        qp = item["qp"]
        obs = observations(item["candidates"])
        risks = risk_scores(obs, qp, risk)
        choices = {arm: obs[select_lowqp(obs, qp, policy, risks, arm)]["name"]
                   for arm in ARMS}
        names = ["identity128"] + [name for name in choices.values() if name != "identity128"]
        streams = {}
        for name in dict.fromkeys(names):
            candidate = variants[name]
            _t, h, w, _ = candidate.shape
            start = time.perf_counter()
            reconstructed, native_bpp = codec._encode_decode_clip(candidate, qp=qp)
            elapsed = time.perf_counter() - start
            prediction, inference_s = timed_inference(analyzer, reconstructed)
            bpp = normalized_bpp(native_bpp, h, w)
            cached = next(row["bpp"] for row in item["candidates"] if row["name"] == name)
            if abs(bpp - cached) > 1e-9:
                raise ValueError(f"V3 bitstream differs from pilot cache: {name}, QP {qp}")
            streams[name] = {"bpp": bpp, "correct": bool(prediction == label),
                "encode_decode_s": elapsed, "inference_s": inference_s}
        result.append({"qp": qp, "choices": choices, "streams": streams})
    return {"sequence_id": meta["sequence_id"], "measurements": result}


def run(args: argparse.Namespace) -> None:
    if not ffmpeg_available():
        raise ValueError("ffmpeg and ffprobe required")
    if kinetics_categories("mc3_18") != kinetics_categories("r3d_18"):
        raise ValueError("mc3 and V2 analyzer categories differ")
    pilot_manifest, risk, frozen, pilot_records = load_pilot(args.archive, args.codec)
    expected_ids = pilot_manifest["split_ids"]["dev"]
    index = json.loads(args.index.read_text(encoding="utf-8"))
    dataset = VideoClipDataset(args.index, split="val", num_frames=16,
                               frame_size=128, temporal_stride=2,
                               train=False, return_metadata=True)
    locations = {clip_id(row): i for i, row in enumerate(dataset.samples)}
    if (len(locations) != len(dataset.samples) or len(expected_ids) != 200
            or any(key not in locations for key in expected_ids)):
        raise ValueError("pilot VAL IDs do not match current Kinetics index")
    positions = list(range(args.shard, len(expected_ids), 2))
    keys = [expected_ids[i] for i in positions]
    if len(keys) != 100:
        raise ValueError("expected 100 videos per shard")
    manifest = {"experiment": "dual_codec_lowqp_v3_mc3_val", "codec": args.codec,
        "shard": args.shard, "shards": 2, "n": len(keys), "sample_ids": keys,
        "sample_fingerprint": digest(keys), "dev_fingerprint": digest(expected_ids),
        "source_archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "pilot_manifest_sha256": digest(pilot_manifest), "risk_sha256": digest(risk),
        "frozen_policy_sha256": digest(frozen), "arms": list(ARMS),
        "index_sha256": hashlib.sha256(args.index.read_bytes()).hexdigest(),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
            cwd=REPO, text=True).strip(), "qps": list(QPS),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__},
        "scope": "Development VAL; old mc3 TEST already informed V3 design."}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("output cache belongs to a different V3 experiment")
    write_json(path, manifest)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analyzer = ActionRecognitionAnalyzer("mc3_18", clip_size=112).freeze().to(device)
    codec = StandardCodec(args.codec, preset="medium", strict_decode=True)
    rows = []
    for number, position in enumerate(positions, 1):
        key = expected_ids[position]
        cache_path = args.out_dir / "cache" / f"clip_{number:04d}.json"
        if cache_path.exists():
            row = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            row = evaluate_clip(dataset, locations[key], pilot_records["dev"][position],
                                risk, frozen["selected_policy"], analyzer, codec)
            write_json(cache_path, row)
        if row["sequence_id"] != key or [m["qp"] for m in row["measurements"]] != list(QPS):
            raise ValueError("stale or incomplete per-video V3 cache")
        rows.append(row)
        print(f"[v3-mc3] {args.codec} shard={args.shard} {number}/100", flush=True)
    record_path = args.out_dir / "shard_records.jsonl"
    temporary = record_path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    temporary.replace(record_path)
    write_json(args.out_dir / "shard_result.json", {"manifest": manifest,
        "diagnostic_only": summarize(rows, args.bootstrap),
        "warning": "Merge two 100-video shards; shard metrics are not final."})


def merge(args: argparse.Namespace) -> None:
    if len(args.shard_dir) != 2:
        raise ValueError("exactly two shard directories required")
    bundles = []
    for directory in args.shard_dir:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        records = [json.loads(line) for line in
                   (directory / "shard_records.jsonl").read_text(encoding="utf-8").splitlines()]
        if (manifest.get("experiment") != "dual_codec_lowqp_v3_mc3_val"
                or manifest.get("codec") != args.codec or len(records) != 100
                or [r["sequence_id"] for r in records] != manifest["sample_ids"]
                or digest(manifest["sample_ids"]) != manifest["sample_fingerprint"]):
            raise ValueError("invalid V3 shard")
        bundles.append((manifest, records))
    bundles.sort(key=lambda pair: pair[0]["shard"])
    if [entry[0]["shard"] for entry in bundles] != [0, 1]:
        raise ValueError("missing or duplicate shard")
    invariants = ("experiment", "codec", "arms", "dev_fingerprint", "qps",
                  "source_archive_sha256", "pilot_manifest_sha256",
                  "risk_sha256", "frozen_policy_sha256", "index_sha256", "code_commit")
    if any(bundles[0][0][key] != bundles[1][0][key] for key in invariants):
        raise ValueError("V3 shard provenance mismatch")
    by_id = {r["sequence_id"]: r for _, rows in bundles for r in rows}
    if len(by_id) != 200:
        raise ValueError("duplicate or missing VAL videos")
    # Reconstruct original interleaved VAL order so every arm is exactly paired.
    ids = [None] * 200
    for manifest, _ in bundles:
        ids[manifest["shard"]::2] = manifest["sample_ids"]
    if digest(ids) != bundles[0][0]["dev_fingerprint"]:
        raise ValueError("merged source fingerprint mismatch")
    rows = [by_id[key] for key in ids]
    write_json(args.out, {"experiment": "dual_codec_lowqp_v3_mc3_val",
        "codec": args.codec, "source_manifests": [m for m, _ in bundles],
        "result": summarize(rows, args.bootstrap),
        "warning": "Development-only; mc3 TEST was examined before V3 design."})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluator = sub.add_parser("evaluate")
    evaluator.add_argument("--archive", type=Path, required=True)
    evaluator.add_argument("--index", type=Path, required=True)
    evaluator.add_argument("--codec", choices=("h264", "h265"), required=True)
    evaluator.add_argument("--shard", type=int, choices=(0, 1), required=True)
    evaluator.add_argument("--out-dir", type=Path, required=True)
    evaluator.add_argument("--bootstrap", type=int, default=200)
    merger = sub.add_parser("merge")
    merger.add_argument("--codec", choices=("h264", "h265"), required=True)
    merger.add_argument("--shard-dir", type=Path, action="append", required=True)
    merger.add_argument("--out", type=Path, required=True)
    merger.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    (run if args.command == "evaluate" else merge)(args)


if __name__ == "__main__":
    main()
