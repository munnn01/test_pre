#!/usr/bin/env python
"""TRAIN-only temporal candidate feasibility against the frozen V2 choice.

The label-aware rescue ceiling is diagnostic, never an inference policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
import torchvision

from ops.dual_codec_lowqp_v3 import load_pilot
from ops.dual_codec_search import digest, write_json
from ops.paper_heldout_mc3 import timed_inference
from ops.rcts_pilot import clip_id
from ops.v6_rate_matched_pilot import CLIPS, MODELS, QPS, SHARDS, choose_qps
from src.codecs.standard import StandardCodec, ffmpeg_available
from src.data.video_dataset import VideoClipDataset
from src.models.codec_search import make_candidates, normalized_bpp
from src.models.dual_codec_search import observations, risk_scores, select_observations
from src.tasks.action_recognition import ActionRecognitionAnalyzer, kinetics_categories

REPO = Path(__file__).resolve().parents[1]
VARIANTS = ("temporal_denoise", "temporal_unsharp")
QP_OFFSETS = (-4, -2, 0, 2, 4)
MATCH_MIN, MATCH_MAX = .95, 1.02


def temporal_variants(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Local temporal residual; never wraps the first/last frame."""
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[0] < 3:
        raise ValueError("expected uint8 T,H,W,3 clip with at least 3 frames")
    x = rgb.astype(np.float32)
    previous = np.concatenate((x[:1], x[:-1]), axis=0)
    following = np.concatenate((x[1:], x[-1:]), axis=0)
    # Suppress temporal mixing across hard frame cuts. The same deterministic
    # gate applies to both transforms and has no access to action labels.
    prev_ok = np.mean(np.abs(x - previous), axis=(1, 2, 3)) < 24
    next_ok = np.mean(np.abs(x - following), axis=(1, 2, 3)) < 24
    weights = np.stack((prev_ok, next_ok), axis=1).astype(np.float32)
    den = 2 + weights.sum(axis=1)[:, None, None, None]
    smooth = (2 * x + weights[:, 0, None, None, None] * previous
              + weights[:, 1, None, None, None] * following) / den
    residual = x - smooth
    return {
        "temporal_denoise": np.clip(np.rint(smooth), 0, 255).astype(np.uint8),
        "temporal_unsharp": np.clip(np.rint(x + .6 * residual), 0, 255).astype(np.uint8),
    }


def predictions(analyzers: dict, video: np.ndarray, label: int) -> tuple[dict, dict]:
    correct, seconds = {}, {}
    for name, analyzer in analyzers.items():
        prediction, elapsed = timed_inference(analyzer, video)
        correct[name] = bool(prediction == label)
        seconds[name] = elapsed
    return correct, seconds


def encode_near(video: np.ndarray, codec: StandardCodec, base_qp: int,
                target_bpp: float) -> tuple[list[tuple[int, np.ndarray, dict]], list[dict]]:
    probes, decoded = {}, {}
    for qp in sorted({base_qp + offset for offset in QP_OFFSETS
                      if 0 <= base_qp + offset <= 51}):
        start = time.perf_counter()
        reconstruction, native_bpp = codec._encode_decode_clip(video, qp=qp)
        _, h, w, _ = video.shape
        probes[qp] = {"qp": qp, "bpp": normalized_bpp(native_bpp, h, w),
                      "encode_decode_s": time.perf_counter() - start}
        decoded[qp] = reconstruction
    ordered = sorted(probes)
    brackets = [(a, b) for a, b in zip(ordered, ordered[1:])
                if (probes[a]["bpp"] - target_bpp)
                   * (probes[b]["bpp"] - target_bpp) <= 0]
    if brackets:
        a, b = min(brackets, key=lambda pair: pair[1] - pair[0])
        for qp in range(a + 1, b):
            start = time.perf_counter()
            reconstruction, native_bpp = codec._encode_decode_clip(video, qp=qp)
            _, h, w, _ = video.shape
            probes[qp] = {"qp": qp, "bpp": normalized_bpp(native_bpp, h, w),
                          "encode_decode_s": time.perf_counter() - start}
            decoded[qp] = reconstruction
    chosen = choose_qps(probes, target_bpp)
    return [(qp, decoded[qp], probes[qp]) for qp in chosen], [probes[qp] for qp in sorted(probes)]


def evaluate_clip(dataset: VideoClipDataset, index: int, cached: dict,
                  analyzers: dict, codec: StandardCodec, old_risk: dict,
                  frozen: dict) -> dict:
    source, label, meta = dataset[index]
    if meta["sequence_id"] != cached["sequence_id"]:
        raise ValueError("V2 cache/video ID mismatch")
    rgb = (source.permute(1, 2, 3, 0).numpy() * 255).round().astype(np.uint8)
    representations = make_candidates(rgb)
    measurements = []
    by_qp = {m["qp"]: m for m in cached["measurements"]}
    for base_qp in QPS:
        previous = by_qp[base_qp]
        obs = observations(previous["candidates"])
        base_index = select_observations(
            obs, base_qp, frozen, risk_scores(obs, base_qp, old_risk))
        base_row = previous["candidates"][base_index]
        base_video = representations[base_row["name"]]
        start = time.perf_counter()
        decoded, native_bpp = codec._encode_decode_clip(base_video, qp=base_qp)
        base_encode_decode_s = time.perf_counter() - start
        _, h, w, _ = base_video.shape
        rate = normalized_bpp(native_bpp, h, w)
        if abs(rate - base_row["bpp"]) > 1e-9:
            raise ValueError("V2 bitstream differs from committed pilot cache")
        mc3_prediction, mc3_seconds = timed_inference(analyzers["mc3_18"], decoded)
        baseline = {"name": base_row["name"], "qp": base_qp, "bpp": rate,
                    "correct": {"r2plus1d_18": bool(base_row["correct"]),
                                "r3d_18": bool(base_row["cross_correct"]),
                                "mc3_18": bool(mc3_prediction == label)},
                    "encode_decode_s": base_encode_decode_s,
                    "mc3_inference_s": mc3_seconds}
        variants = temporal_variants(base_video)
        candidates, searches, clean = [], {}, {}
        for name, video in variants.items():
            clean[name], _ = predictions(analyzers, video, label)
            shortlisted, probes = encode_near(video, codec, base_qp, rate)
            searches[name] = probes
            for qp, reconstruction, probe in shortlisted:
                correct, seconds = predictions(analyzers, reconstruction, label)
                candidates.append({"name": name, "qp": qp,
                    "bpp": probe["bpp"], "rate_ratio_to_v2": probe["bpp"] / rate,
                    "correct": correct, "inference_s": seconds,
                    "encode_decode_s": probe["encode_decode_s"]})
        measurements.append({"qp": base_qp, "v2": baseline,
                             "clean_correct": clean, "candidates": candidates,
                             "search": searches})
    return {"schema": 1, "sequence_id": meta["sequence_id"],
            "measurements": measurements}


def summarize(rows: list[dict]) -> dict:
    out = {"n_videos": len(rows), "qps": {},
           "scope": "TRAIN-fit label-aware feasibility, not deployable performance"}
    for qp in QPS:
        cells = [next(m for m in row["measurements"] if m["qp"] == qp)
                 for row in rows]
        report = {"n": len(cells), "v2_mc3_wrong": 0,
                  "matched_any": 0, "rescue_matched": 0,
                  "new_mc3_error_possible": 0,
                  "matched_by_variant": {name: 0 for name in VARIANTS},
                  "rescue_by_variant": {name: 0 for name in VARIANTS},
                  "search_encodes": 0, "search_encode_decode_s": 0.,
                  "evaluated_candidates": 0}
        for m in cells:
            base = m["v2"]
            report["v2_mc3_wrong"] += int(not base["correct"]["mc3_18"])
            report["search_encodes"] += sum(len(v) for v in m["search"].values())
            report["search_encode_decode_s"] += sum(
                p["encode_decode_s"] for v in m["search"].values() for p in v)
            report["evaluated_candidates"] += len(m["candidates"])
            matched = [c for c in m["candidates"]
                       if MATCH_MIN <= c["rate_ratio_to_v2"] <= MATCH_MAX]
            report["matched_any"] += int(bool(matched))
            report["new_mc3_error_possible"] += int(
                base["correct"]["mc3_18"] and
                any(not c["correct"]["mc3_18"] for c in matched))
            safe = [c for c in matched if not base["correct"]["mc3_18"]
                    and c["correct"]["mc3_18"]
                    and all(c["correct"][model] >= base["correct"][model]
                            for model in MODELS[:2])]
            report["rescue_matched"] += int(bool(safe))
            for name in VARIANTS:
                report["matched_by_variant"][name] += int(
                    any(c["name"] == name for c in matched))
                report["rescue_by_variant"][name] += int(
                    any(c["name"] == name for c in safe))
        out["qps"][str(qp)] = report
    return out


def run(args: argparse.Namespace) -> None:
    if not ffmpeg_available():
        raise ValueError("ffmpeg/ffprobe required")
    if any(kinetics_categories(model) != kinetics_categories(MODELS[0])
           for model in MODELS[1:]):
        raise ValueError("analyzer category maps differ")
    pilot_manifest, old_risk, old_frozen, pilot = load_pilot(
        args.archive, args.codec, stages=("fit",))
    expected_ids = pilot_manifest["split_ids"]["fit"][:CLIPS]
    if len(expected_ids) != CLIPS:
        raise ValueError("pilot FIT source list is incomplete")
    positions = list(range(args.shard, CLIPS, SHARDS))
    dataset = VideoClipDataset(args.index, split="train", num_frames=16,
                               frame_size=128, temporal_stride=2,
                               train=False, return_metadata=True)
    locations = {clip_id(row): i for i, row in enumerate(dataset.samples)}
    if len(locations) != len(dataset.samples) or any(key not in locations for key in expected_ids):
        raise ValueError("TRAIN source list mismatch")
    keys = [expected_ids[i] for i in positions]
    manifest = {"experiment": "v7_temporal_train_oracle", "codec": args.codec,
        "shard": args.shard, "shards": SHARDS, "qps": list(QPS),
        "variants": list(VARIANTS), "qp_offsets": list(QP_OFFSETS),
        "rate_match_interval": [MATCH_MIN, MATCH_MAX],
        "sample_ids": keys, "sample_fingerprint": digest(keys),
        "full_sample_fingerprint": digest(expected_ids),
        "pilot_manifest_sha256": digest(pilot_manifest),
        "pilot_archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "v2_risk_sha256": digest(old_risk),
        "v2_frozen_sha256": digest(old_frozen),
        "index_sha256": hashlib.sha256(args.index.read_bytes()).hexdigest(),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
             cwd=REPO, text=True).strip(),
        "versions": {"python": platform.python_version(),
                     "torch": torch.__version__, "torchvision": torchvision.__version__},
        "scope": "First 30 frozen V2 TRAIN-fit IDs; label-aware feasibility only."}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("stale V7 output directory")
    write_json(manifest_path, manifest)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analyzers = {name: ActionRecognitionAnalyzer(name, clip_size=112).freeze().to(device)
                 for name in MODELS}
    codec = StandardCodec(args.codec, preset="medium", strict_decode=True)
    rows = []
    for number, position in enumerate(positions, 1):
        key = expected_ids[position]
        path = args.out_dir / "cache" / f"clip_{number:04d}.json"
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
        else:
            row = evaluate_clip(dataset, locations[key], pilot["fit"][position],
                                analyzers, codec, old_risk,
                                old_frozen["selected_policy"])
            write_json(path, row)
        if (row.get("schema") != 1 or row.get("sequence_id") != key
                or [m["qp"] for m in row["measurements"]] != list(QPS)):
            raise ValueError("stale/incomplete per-clip V7 cache")
        rows.append(row)
        print(f"[v7] {args.codec} shard={args.shard} {number}/{len(positions)}", flush=True)
    out = args.out_dir / "shard_records.jsonl"
    temporary = out.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    temporary.replace(out)
    write_json(args.out_dir / "shard_result.json", {"manifest": manifest,
        "diagnostic_only": summarize(rows),
        "warning": "TRAIN label-aware oracle is not achieved policy performance."})


def merge(args: argparse.Namespace) -> None:
    if len(args.shard_dir) != SHARDS:
        raise ValueError("exactly two shard directories required")
    bundles = []
    for directory in args.shard_dir:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in
                (directory / "shard_records.jsonl").read_text(encoding="utf-8").splitlines()]
        if (manifest.get("experiment") != "v7_temporal_train_oracle"
                or manifest.get("codec") != args.codec or len(rows) != CLIPS // SHARDS
                or [r["sequence_id"] for r in rows] != manifest["sample_ids"]
                or digest(manifest["sample_ids"]) != manifest["sample_fingerprint"]):
            raise ValueError("invalid V7 shard")
        bundles.append((manifest, rows))
    bundles.sort(key=lambda bundle: bundle[0]["shard"])
    if [bundle[0]["shard"] for bundle in bundles] != list(range(SHARDS)):
        raise ValueError("missing/duplicate V7 shard")
    common = ("experiment", "codec", "shards", "qps", "variants", "qp_offsets",
              "rate_match_interval", "full_sample_fingerprint",
              "pilot_manifest_sha256", "pilot_archive_sha256", "v2_risk_sha256",
              "v2_frozen_sha256", "code_commit")
    if any(bundles[0][0][key] != bundles[1][0][key] for key in common):
        raise ValueError("V7 shard provenance differs")
    ids = [None] * CLIPS
    by_id = {}
    for manifest, rows in bundles:
        ids[manifest["shard"]::SHARDS] = manifest["sample_ids"]
        by_id.update({r["sequence_id"]: r for r in rows})
    if len(by_id) != CLIPS or digest(ids) != bundles[0][0]["full_sample_fingerprint"]:
        raise ValueError("V7 merge source fingerprint differs")
    report = {"experiment": "v7_temporal_train_oracle", "codec": args.codec,
              "source_manifests": [manifest for manifest, _ in bundles],
              "result": summarize([by_id[key] for key in ids])}
    write_json(args.out, report)
    print(json.dumps({"codec": args.codec, "result": report["result"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluator = commands.add_parser("evaluate")
    evaluator.add_argument("--archive", type=Path, required=True)
    evaluator.add_argument("--index", type=Path, required=True)
    evaluator.add_argument("--codec", choices=("h264", "h265"), required=True)
    evaluator.add_argument("--shard", type=int, choices=(0, 1), required=True)
    evaluator.add_argument("--out-dir", type=Path, required=True)
    merged = commands.add_parser("merge")
    merged.add_argument("--codec", choices=("h264", "h265"), required=True)
    merged.add_argument("--shard-dir", type=Path, action="append", required=True)
    merged.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    (run if args.command == "evaluate" else merge)(args)


if __name__ == "__main__":
    main()
