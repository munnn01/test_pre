#!/usr/bin/env python
"""Paired five-QP V8 evaluation of one TRAIN-calibrated H.264 restorer."""
from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import tarfile
import time

import numpy as np
import torch

from ops.codec_search_ar import QPS, as_video
from ops.dual_codec_search import digest, write_json
from ops.dual_codec_search_confirm_1000 import CONFIG, load_frozen, sample_plan
from ops.merge_dual_codec_search_confirm_1000 import fingerprint, load_shards
from ops.rcts_pilot import clip_id
from src.codecs.standard import StandardCodec, ffmpeg_available
from src.metrics.bd_rate import bd_metric, bd_rate
from src.models.codec_search import make_candidates, normalized_bpp
from src.models.dual_codec_search import MODELS as TRAIN_MODELS, select
from src.models.v8_restorer import V8Restorer
from src.tasks.action_recognition import ActionRecognitionAnalyzer, kinetics_categories

REPO = Path(__file__).resolve().parents[1]
MODELS = (*TRAIN_MODELS, "mc3_18")
ARMS = ("codec", "v2", "codec_restored", "v2_restored")
COMPARISONS = (("v2", "codec"), ("codec_restored", "codec"),
               ("v2_restored", "codec"), ("v2_restored", "codec_restored"))


def checkpoint_bundle(archive: Path) -> tuple[dict, dict, V8Restorer | None]:
    prefix = "outputs/v8_restorer/h264/train/"
    with tarfile.open(archive, "r:gz") as bundle:
        def read(name: str, limit: int) -> bytes:
            member = bundle.getmember(prefix + name)
            if not member.isfile() or not 0 < member.size <= limit:
                raise ValueError(f"invalid V8 checkpoint member: {name}")
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError(f"unreadable checkpoint member: {name}")
            return stream.read()
        manifest = json.loads(read("manifest.json", 1_000_000))
        calibration = json.loads(read("calibration.json", 1_000_000))
        if (manifest["experiment"] != "v8_shared_decoder_h264"
                or manifest["teachers"] != list(TRAIN_MODELS)
                or manifest["excluded_analyzer"] != "mc3_18"
                or calibration["manifest_sha256"] != digest(manifest)
                or calibration["mc3_access"] is not False
                or calibration["selected"] not in ("pixel", "semantic", "v2")):
            raise ValueError("V8 calibration provenance invalid")
        model = None
        if calibration["selected"] != "v2":
            payload = torch.load(BytesIO(read(calibration["selected"] + ".pth", 50_000_000)),
                                 map_location="cpu", weights_only=True)
            if (payload["schema"] != 1 or
                    payload["metadata"]["manifest_sha256"] != digest(manifest)):
                raise ValueError("selected V8 checkpoint differs from training manifest")
            model = V8Restorer()
            model.load_state_dict(payload["model"], strict=True)
    return manifest, calibration, model


def report_curves(rows: list[dict]) -> dict:
    curves = {}
    for model in MODELS:
        curves[model] = {}
        for arm in ARMS:
            by_qp = {}
            for qp in QPS:
                points = [next(m["arms"][arm] for m in row["measurements"]
                               if m["qp"] == qp) for row in rows]
                by_qp[str(qp)] = {"n": len(points),
                    "bpp": float(np.mean([p["bpp"] for p in points])),
                    "top1": float(np.mean([p["correct"][model] for p in points]))}
            curves[model][arm] = by_qp
    return curves


def compare_curves(curves: dict) -> dict:
    def finite(value: float) -> float | None:
        return float(value) if math.isfinite(value) else None

    out = {}
    for model, arms in curves.items():
        out[model] = {}
        for trial, anchor in COMPARISONS:
            a, b = [arms[anchor][str(q)] for q in QPS], [arms[trial][str(q)] for q in QPS]
            out[model][f"{trial}_vs_{anchor}"] = {
                "bd_rate_top1_pct": finite(bd_rate(
                    [p["bpp"] for p in a], [p["top1"] for p in a],
                    [p["bpp"] for p in b], [p["top1"] for p in b])),
                "bd_accuracy_top1_pp": finite(100 * bd_metric(
                    [p["bpp"] for p in a], [p["top1"] for p in a],
                    [p["bpp"] for p in b], [p["top1"] for p in b])),
                "min_same_qp_top1_gap_pp": 100 * min(x["top1"] - y["top1"]
                                                   for x, y in zip(b, a))}
    return out


def summarize(rows: list[dict], bootstrap: int) -> dict:
    curves = report_curves(rows)
    point = compare_curves(curves)
    intervals = None
    if bootstrap:
        rng = np.random.default_rng(20260927)
        samples = {model: {arm: {"bd_rate_top1_pct": [], "bd_accuracy_top1_pp": []}
                           for arm in point[model]} for model in MODELS}
        for _ in range(bootstrap):
            picked = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
            draw = compare_curves(report_curves(picked))
            for model in MODELS:
                for arm in draw[model]:
                    for metric in samples[model][arm]:
                        value = draw[model][arm][metric]
                        if value is not None:
                            samples[model][arm][metric].append(value)
        intervals = {model: {arm: {metric: {
            "valid_draws": len(values),
            "ci95": np.percentile(values, [2.5, 97.5]).tolist()
                    if values else None}
            for metric, values in samples[model][arm].items()}
            for arm in samples[model]} for model in MODELS}
    runtime = {}
    for arm in ARMS:
        cells = [m["arms"][arm] for row in rows for m in row["measurements"]]
        runtime[arm] = {
            "mean_restoration_s_per_16_frames": float(np.mean(
                [cell["restoration_s"] for cell in cells])),
            "mean_encode_decode_s_per_16_frames": float(np.mean(
                [cell["encode_decode_s"] for cell in cells])),
            "mean_analyzer_s_per_16_frames": float(np.mean(
                [cell["analyzer_s"] for cell in cells])),
            "timing_scope": "selected-bitstream encode/decode and three-analyzer inference; "
                            "six-candidate selection cost is separate"}
    return {"n": len(rows), "curves": curves, "metrics": point,
            "bootstrap": intervals, "runtime": runtime,
            "bootstrap_unit": "whole clip with all five QPs paired"}


@torch.no_grad()
def evaluate_clip(dataset, index: int, cached: dict, codec: StandardCodec,
                  risk: dict, policy: dict, restorer: V8Restorer | None,
                  analyzers: dict, device: torch.device) -> dict:
    source, label, meta = dataset[index]
    if meta["sequence_id"] != cached["sequence_id"]:
        raise ValueError("V2 cache/decoded source ID mismatch")
    rgb = (source.permute(1, 2, 3, 0).numpy() * 255).round().astype(np.uint8)
    candidates = make_candidates(rgb)
    measurements = []
    for previous in cached["measurements"]:
        qp = previous["qp"]
        chosen = previous["candidates"][select(previous["candidates"], qp, policy, risk)]
        outputs = {}
        for role, item in (("codec", previous["candidates"][0]), ("v2", chosen)):
            name = item["name"]
            if name in outputs:
                outputs[role] = outputs[name]
                continue
            video = candidates[name]
            native_size = video.shape[1]
            if device.type == "cuda":
                torch.cuda.synchronize()
            encode_start = time.perf_counter()
            decoded, native_bpp = codec._encode_decode_clip(video, qp=qp)
            if device.type == "cuda":
                torch.cuda.synchronize()
            encode_decode_s = time.perf_counter() - encode_start
            bpp = normalized_bpp(native_bpp, native_size, native_size)
            if abs(bpp - item["bpp"]) > 1e-9:
                raise ValueError("selected/anchor bitstream differs from frozen V2 TEST cache")
            outputs[name] = (as_video(decoded).to(device), bpp, native_size,
                             encode_decode_s)
            outputs[role] = outputs[name]
        arms = {}
        for role in ("codec", "v2"):
            decoded, bpp, native_size, encode_decode_s = outputs[role]
            for enhanced in (False, True):
                arm = role + ("_restored" if enhanced else "")
                start = time.perf_counter()
                if enhanced and restorer is not None:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    video = restorer(decoded, qp, native_size)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    restoration_s = time.perf_counter() - start
                else:
                    video, restoration_s = decoded, 0.
                if device.type == "cuda":
                    torch.cuda.synchronize()
                analyzer_start = time.perf_counter()
                correct = {model: bool(analyzer.predict(video).argmax(1).item() == label)
                           for model, analyzer in analyzers.items()}
                if device.type == "cuda":
                    torch.cuda.synchronize()
                analyzer_s = time.perf_counter() - analyzer_start
                if not enhanced:
                    cached_row = previous["candidates"][0 if role == "codec" else
                          select(previous["candidates"], qp, policy, risk)]
                    if (correct[TRAIN_MODELS[0]] != cached_row["correct"]
                            or correct[TRAIN_MODELS[1]] != cached_row["cross_correct"]):
                        raise ValueError("known-analyzer V2 cache outcome differs")
                arms[arm] = {"bpp": bpp, "correct": correct,
                             "restoration_s": restoration_s,
                             "encode_decode_s": encode_decode_s,
                             "analyzer_s": analyzer_s,
                             "candidate": "identity128" if role == "codec" else chosen["name"]}
        measurements.append({"qp": qp, "arms": arms})
    return {"schema": 1, "sequence_id": meta["sequence_id"],
            "measurements": measurements}


def run(args: argparse.Namespace) -> None:
    if not ffmpeg_available():
        raise ValueError("ffmpeg/ffprobe required")
    if any(kinetics_categories(model) != kinetics_categories(MODELS[0])
           for model in MODELS[1:]):
        raise ValueError("analyzer class orders differ")
    torch.set_num_threads(2)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    cached, source_manifests = load_shards(args.cache_dir, "h264", config)
    by_id = {row["sequence_id"]: row for row in cached}
    risk, frozen, policy = load_frozen("h264", config)
    train_manifest, calibration, restorer = checkpoint_bundle(args.checkpoint_archive)
    if (train_manifest["frozen_policy_sha256"] != digest(frozen)
            or train_manifest["risk_sha256"] != digest(risk)):
        raise ValueError("V8 restorer trained on a different V2 policy")
    if train_manifest["codec"] != "h264":
        raise ValueError("V8 restorer is not the H.264 model")
    dataset, all_indices = sample_plan(args.index, config)
    indices = [i for pos, i in enumerate(all_indices) if pos % 2 == args.shard]
    if len(indices) != 500 or any(clip_id(dataset.samples[i]) not in by_id for i in indices):
        raise ValueError("incomplete V2 TEST cache for shard")
    manifest = {"experiment": "v8_restorer_h264_test", "shard": args.shard,
        "n": len(indices), "sample_ids": [clip_id(dataset.samples[i]) for i in indices],
        "shard_fingerprint": fingerprint([clip_id(dataset.samples[i]) for i in indices]),
        "test_fingerprint": config["test_fingerprint"], "qps": list(QPS),
        "models": list(MODELS), "arms": list(ARMS),
        "selected": calibration["selected"],
        "checkpoint_archive_sha256": hashlib.sha256(args.checkpoint_archive.read_bytes()).hexdigest(),
        "train_manifest_sha256": digest(train_manifest),
        "calibration_sha256": digest(calibration),
        "source_cache_manifest_sha256": [digest(m) for m in source_manifests],
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
             cwd=REPO, text=True).strip(),
        "torch": torch.__version__, "python": platform.python_version(),
        "scope": "historically inspected 1,000 TEST clips; V8 locked by TRAIN-cal only"}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("stale V8 evaluation directory")
    write_json(path, manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if restorer is not None:
        restorer = restorer.to(device).eval()
    analyzers = {name: ActionRecognitionAnalyzer(name, clip_size=112).freeze().to(device)
                 for name in MODELS}
    codec = StandardCodec("h264", preset="medium", strict_decode=True)
    rows = []
    for number, index in enumerate(indices, 1):
        key = clip_id(dataset.samples[index])
        cache = args.out_dir / "cache" / f"clip_{number:04d}.json"
        if cache.exists():
            row = json.loads(cache.read_text(encoding="utf-8"))
        else:
            row = evaluate_clip(dataset, index, by_id[key], codec, risk, policy,
                                restorer, analyzers, device)
            write_json(cache, row)
        if (row["sequence_id"] != key or
                [m["qp"] for m in row["measurements"]] != list(QPS)):
            raise ValueError("stale/incomplete V8 per-clip evaluation cache")
        rows.append(row)
        if number % 10 == 0 or number == len(indices):
            print(f"[v8-eval] shard={args.shard} {number}/{len(indices)}", flush=True)
    records = args.out_dir / "shard_records.jsonl"
    temporary = records.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows),
                         encoding="utf-8")
    temporary.replace(records)
    write_json(args.out_dir / "shard_result.json", {"manifest": manifest,
        "diagnostic_only": summarize(rows, 0),
        "max_resident_gpu_bytes_including_analyzers":
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None,
        "warning": "Merge two disjoint 500-clip shards before BD-rate reporting."})


def merge(args: argparse.Namespace) -> None:
    if len(args.shard_dir) != 2:
        raise ValueError("two shards required")
    bundles = []
    for directory in args.shard_dir:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        records = [json.loads(line) for line in
                   (directory / "shard_records.jsonl").read_text(encoding="utf-8").splitlines()]
        if (manifest["experiment"] != "v8_restorer_h264_test" or len(records) != 500
                or [r["sequence_id"] for r in records] != manifest["sample_ids"]
                or fingerprint(manifest["sample_ids"]) != manifest["shard_fingerprint"]):
            raise ValueError("invalid V8 evaluation shard")
        bundles.append((manifest, records))
    bundles.sort(key=lambda item: item[0]["shard"])
    if [m["shard"] for m, _ in bundles] != [0, 1]:
        raise ValueError("missing or duplicate V8 shard")
    common = ("experiment", "test_fingerprint", "qps", "models", "arms",
              "selected", "checkpoint_archive_sha256", "train_manifest_sha256",
              "calibration_sha256", "source_cache_manifest_sha256", "code_commit")
    if any(bundles[0][0][key] != bundles[1][0][key] for key in common):
        raise ValueError("V8 shards differ in provenance")
    rows = bundles[0][1] + bundles[1][1]
    if (len({r["sequence_id"] for r in rows}) != 1000
            or fingerprint([r["sequence_id"] for r in rows]) !=
            bundles[0][0]["test_fingerprint"]):
        raise ValueError("V8 merged 1,000-clip fingerprint mismatch")
    report = {"experiment": "v8_restorer_h264_test", "n": len(rows),
        "scope": "paired replication on historically inspected TEST",
        "source_manifests": [m for m, _ in bundles],
        "result": summarize(rows, args.bootstrap)}
    write_json(args.out, report)
    print(json.dumps({"selected": bundles[0][0]["selected"],
                      "metrics": report["result"]["metrics"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--index", type=Path, required=True)
    evaluate.add_argument("--checkpoint-archive", type=Path, required=True)
    evaluate.add_argument("--cache-dir", type=Path, action="append", required=True)
    evaluate.add_argument("--shard", type=int, choices=(0, 1), required=True)
    evaluate.add_argument("--out-dir", type=Path, required=True)
    merged = sub.add_parser("merge")
    merged.add_argument("--shard-dir", type=Path, action="append", required=True)
    merged.add_argument("--out", type=Path, required=True)
    merged.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    (run if args.command == "evaluate" else merge)(args)


if __name__ == "__main__":
    main()
