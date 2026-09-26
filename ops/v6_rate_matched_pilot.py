#!/usr/bin/env python
"""TRAIN-only resolution/QP feasibility pilot against the frozen V2 choice.

The label-aware oracle in the summary is diagnostic and never an inference
policy. Every candidate uses the real x264/x265 encoder and one shared decoded
video for three frozen action-recognition analyzers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

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
from src.models.codec_search import make_candidates, normalized_bpp
from src.models.dual_codec_search import observations, risk_scores, select_observations
from src.tasks.action_recognition import ActionRecognitionAnalyzer, kinetics_categories

REPO = Path(__file__).resolve().parents[1]
QPS = (30, 35, 40)
MODELS = ("r2plus1d_18", "r3d_18", "mc3_18")
RESOLUTIONS = (96, 112, 128)
QP_OFFSETS = (-4, 0, 4, 8, 12, 16)
MATCH_MIN, MATCH_MAX = .95, 1.02
NEAR_MAX = 1.05
CLIPS = 30
SHARDS = 2


def qp_grid(base_qp: int) -> tuple[int, ...]:
    return tuple(sorted({base_qp + offset for offset in QP_OFFSETS
                         if 0 <= base_qp + offset <= 51}))


def candidate_names(rgb: np.ndarray) -> dict[int, np.ndarray]:
    variants = make_candidates(rgb)
    return {96: variants["area96"], 112: variants["area112"],
            128: variants["identity128"]}


def _predict_all(analyzers: dict, video: np.ndarray, label: int) -> tuple[dict, dict]:
    correct, seconds = {}, {}
    for name, analyzer in analyzers.items():
        prediction, elapsed = timed_inference(analyzer, video)
        correct[name] = bool(prediction == label)
        seconds[name] = elapsed
    return correct, seconds


def choose_qps(probes: dict[int, dict], target: float) -> list[int]:
    """One best under budget and, if close enough, the nearest measured QP."""
    if target <= 0:
        raise ValueError("invalid V2 bitrate target")
    under = [qp for qp, item in probes.items()
             if item["bpp"] <= target * MATCH_MAX + 1e-12]
    choices = set()
    if under:
        choices.add(min(under, key=lambda qp: abs(np.log(probes[qp]["bpp"] / target))))
    nearest = min(probes, key=lambda qp: abs(np.log(probes[qp]["bpp"] / target)))
    if probes[nearest]["bpp"] <= target * NEAR_MAX + 1e-12:
        choices.add(nearest)
    return sorted(choices)


def encode_grid(video: np.ndarray, codec: StandardCodec, base_qp: int,
                target_bpp: float) -> tuple[dict[int, tuple[np.ndarray, dict]], list[dict]]:
    """Measure real bytes first; retain decoded frames only for shortlisted QPs."""
    outputs = {}
    probes = {}
    for qp in qp_grid(base_qp):
        start = time.perf_counter()
        reconstructed, native_bpp = codec._encode_decode_clip(video, qp=qp)
        _, h, w, _ = video.shape
        rate = normalized_bpp(native_bpp, h, w)
        probes[qp] = {"qp": qp, "bpp": rate,
                      "encode_decode_s": time.perf_counter() - start}
        outputs[qp] = reconstructed
    # Probe integer QPs around the first measured rate crossing. This reduces
    # missed matches caused by a coarse four-QP grid without assuming that
    # measured codec rates are perfectly monotone.
    ordered = sorted(probes)
    brackets = [(a, b) for a, b in zip(ordered, ordered[1:])
                if (probes[a]["bpp"] - target_bpp)
                   * (probes[b]["bpp"] - target_bpp) <= 0]
    if brackets:
        a, b = min(brackets, key=lambda pair: pair[1] - pair[0])
        for qp in range(a + 1, b):
            start = time.perf_counter()
            reconstructed, native_bpp = codec._encode_decode_clip(video, qp=qp)
            _, h, w, _ = video.shape
            probes[qp] = {"qp": qp, "bpp": normalized_bpp(native_bpp, h, w),
                          "encode_decode_s": time.perf_counter() - start}
            outputs[qp] = reconstructed
    selected = choose_qps(probes, target_bpp)
    return {qp: (outputs[qp], probes[qp]) for qp in selected}, [probes[qp] for qp in sorted(probes)]


def evaluate_clip(dataset: VideoClipDataset, index: int, cached: dict,
                  analyzers: dict, codec: StandardCodec,
                  old_risk: dict, frozen: dict) -> dict:
    source, label, meta = dataset[index]
    if meta["sequence_id"] != cached["sequence_id"]:
        raise ValueError("V2 cache/video ID mismatch")
    capture = cv2.VideoCapture(dataset.samples[index]["path"])
    ok, _ = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"undecodable source: {meta['sequence_id']}")
    rgb = (source.permute(1, 2, 3, 0).numpy() * 255).round().astype(np.uint8)
    representations = candidate_names(rgb)
    clean = {}
    for resolution, video in representations.items():
        correct, _seconds = _predict_all(analyzers, video, label)
        clean[str(resolution)] = correct
    measurements = []
    by_qp = {m["qp"]: m for m in cached["measurements"]}
    for base_qp in QPS:
        previous = by_qp[base_qp]
        obs = observations(previous["candidates"])
        base_index = select_observations(
            obs, base_qp, frozen, risk_scores(obs, base_qp, old_risk))
        base_row = previous["candidates"][base_index]
        base_video = make_candidates(rgb)[base_row["name"]]
        base_start = time.perf_counter()
        reconstructed, native_bpp = codec._encode_decode_clip(base_video, qp=base_qp)
        base_encode_decode_s = time.perf_counter() - base_start
        _, bh, bw, _ = base_video.shape
        measured = normalized_bpp(native_bpp, bh, bw)
        if abs(measured - base_row["bpp"]) > 1e-9:
            raise ValueError("V2 bitstream differs from committed pilot cache")
        mc3_prediction, base_inference_s = timed_inference(analyzers["mc3_18"], reconstructed)
        baseline = {"name": base_row["name"], "qp": base_qp,
            "bpp": measured, "correct": {
                "r2plus1d_18": bool(base_row["correct"]),
                "r3d_18": bool(base_row["cross_correct"]),
                "mc3_18": bool(mc3_prediction == label)},
            "mc3_inference_s": base_inference_s,
            "encode_decode_s": base_encode_decode_s}
        candidates, search = [], {}
        for resolution, video in representations.items():
            shortlisted, probes = encode_grid(video, codec, base_qp, measured)
            search[str(resolution)] = probes
            for qp, (decoded, probe) in shortlisted.items():
                rate = probe["bpp"]
                correct, seconds = _predict_all(analyzers, decoded, label)
                candidates.append({"resolution": resolution, "qp": qp,
                    "bpp": rate, "rate_ratio_to_v2": rate / measured,
                    "correct": correct, "inference_s": seconds,
                    "encode_decode_s": probe["encode_decode_s"]})
        measurements.append({"qp": base_qp, "v2": baseline,
                             "candidates": candidates, "search": search})
    return {"schema": 1, "sequence_id": meta["sequence_id"],
            "clean_correct": clean, "measurements": measurements}


def summarize(rows: list[dict]) -> dict:
    points = [m for row in rows for m in row["measurements"]]
    out = {"n_videos": len(rows), "n_qp_video_points": len(points),
           "qps": {}, "scope": "TRAIN-fit oracle feasibility only; reads labels"}
    for qp in QPS:
        cells = [m for m in points if m["qp"] == qp]
        result = {"n": len(cells), "v2_mc3_wrong": 0,
                  "matched_any": 0, "rescue_matched": 0,
                  "rescue_under_budget": 0,
                  "matched_by_resolution": {str(r): 0 for r in RESOLUTIONS},
                  "rescue_by_resolution": {str(r): 0 for r in RESOLUTIONS},
                  "oracle_mean_bpp_change_pct": 0.,
                  "search_encodes": 0, "search_encode_decode_s": 0.,
                  "evaluated_candidates": 0}
        if not cells:
            out["qps"][str(qp)] = result
            continue
        total_base_rate, total_oracle_rate = 0., 0.
        for cell in cells:
            base = cell["v2"]
            result["search_encodes"] += sum(len(probes) for probes in cell["search"].values())
            result["search_encode_decode_s"] += sum(
                p["encode_decode_s"] for probes in cell["search"].values() for p in probes)
            result["evaluated_candidates"] += len(cell["candidates"])
            total_base_rate += base["bpp"]
            total_oracle_rate += base["bpp"]
            result["v2_mc3_wrong"] += int(not base["correct"]["mc3_18"])
            matched = [c for c in cell["candidates"]
                       if MATCH_MIN <= c["rate_ratio_to_v2"] <= MATCH_MAX]
            result["matched_any"] += int(bool(matched))
            for resolution in RESOLUTIONS:
                result["matched_by_resolution"][str(resolution)] += int(
                    any(c["resolution"] == resolution for c in matched))
            def rescue(c: dict) -> bool:
                return (not base["correct"]["mc3_18"]
                        and c["correct"]["mc3_18"]
                        and all(c["correct"][model] >= base["correct"][model]
                                for model in MODELS[:2]))
            safe_matched = [c for c in matched if rescue(c)]
            safe_budget = [c for c in cell["candidates"]
                           if c["rate_ratio_to_v2"] <= MATCH_MAX and rescue(c)]
            result["rescue_matched"] += int(bool(safe_matched))
            result["rescue_under_budget"] += int(bool(safe_budget))
            for resolution in RESOLUTIONS:
                result["rescue_by_resolution"][str(resolution)] += int(
                    any(c["resolution"] == resolution for c in safe_matched))
            if safe_matched:
                cheapest = min(safe_matched, key=lambda c: c["bpp"])
                total_oracle_rate += cheapest["bpp"] - base["bpp"]
        result["oracle_mean_bpp_change_pct"] = 100 * (total_oracle_rate / total_base_rate - 1)
        out["qps"][str(qp)] = result
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
    manifest = {"experiment": "v6_resolution_qp_train_oracle", "codec": args.codec,
        "shard": args.shard, "shards": SHARDS, "qps": list(QPS),
        "resolutions": list(RESOLUTIONS), "qp_offsets": list(QP_OFFSETS),
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
        "scope": "First 30 frozen V2 TRAIN-fit IDs; label-aware feasibility, no policy fit."}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("stale V6 output directory")
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
            raise ValueError("stale/incomplete per-clip V6 cache")
        rows.append(row)
        print(f"[v6] {args.codec} shard={args.shard} {number}/{len(positions)}", flush=True)
    out = args.out_dir / "shard_records.jsonl"
    temporary = out.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    temporary.replace(out)
    write_json(args.out_dir / "shard_result.json", {"manifest": manifest,
        "diagnostic_only": summarize(rows),
        "warning": "TRAIN-fit label-aware oracle is an opportunity bound, not achieved policy performance."})


def merge(args: argparse.Namespace) -> None:
    if len(args.shard_dir) != SHARDS:
        raise ValueError("exactly two shard directories required")
    bundles = []
    for directory in args.shard_dir:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in
                (directory / "shard_records.jsonl").read_text(encoding="utf-8").splitlines()]
        if (manifest.get("experiment") != "v6_resolution_qp_train_oracle"
                or manifest.get("codec") != args.codec or len(rows) != CLIPS // SHARDS
                or [r["sequence_id"] for r in rows] != manifest["sample_ids"]
                or digest(manifest["sample_ids"]) != manifest["sample_fingerprint"]):
            raise ValueError("invalid V6 shard")
        bundles.append((manifest, rows))
    bundles.sort(key=lambda bundle: bundle[0]["shard"])
    if [bundle[0]["shard"] for bundle in bundles] != list(range(SHARDS)):
        raise ValueError("missing/duplicate V6 shard")
    common = ("experiment", "codec", "shards", "qps", "resolutions",
              "qp_offsets", "rate_match_interval", "full_sample_fingerprint",
              "pilot_manifest_sha256", "pilot_archive_sha256", "v2_risk_sha256",
              "v2_frozen_sha256", "code_commit")
    if any(bundles[0][0][key] != bundles[1][0][key] for key in common):
        raise ValueError("V6 shard provenance differs")
    ids = [None] * CLIPS
    by_id = {}
    for manifest, rows in bundles:
        ids[manifest["shard"]::SHARDS] = manifest["sample_ids"]
        by_id.update({r["sequence_id"]: r for r in rows})
    if len(by_id) != CLIPS or digest(ids) != bundles[0][0]["full_sample_fingerprint"]:
        raise ValueError("V6 merge source fingerprint differs")
    report = {"experiment": "v6_resolution_qp_train_oracle", "codec": args.codec,
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
