#!/usr/bin/env python
"""Time the full six-candidate V2 search against codec-only encoding.

Unlike paper_heldout_mc3, this benchmark includes every trial encode/decode
and both frozen analyzer passes needed by the selector. It excludes model
weight download and process startup, and uses a fixed, documented sample.
The codec-only arm does not run task analyzers: they are not part of its encoder.
FFmpeg encode+decode is measured in both arms, so this is not encode-only latency.
"""
from __future__ import annotations

import argparse
import hashlib
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

import numpy as np
import torch
import torchvision

from ops.codec_search_ar import QPS, as_video, predict_and_feature
from ops.dual_codec_search import digest, signals, write_json
from ops.dual_codec_search_confirm_1000 import CONFIG, file_sha256, load_frozen
from ops.rcts_pilot import balanced_indices, clip_id
from src.codecs.standard import StandardCodec, ffmpeg_available
from src.data.video_dataset import VideoClipDataset
from src.models.codec_search import CANDIDATES, make_candidates, normalized_bpp
from src.models.dual_codec_search import MODELS, select
from src.tasks.action_recognition import ActionRecognitionAnalyzer


def measured_prediction(analyzer, video):
    if next(analyzer.parameters()).is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = predict_and_feature(analyzer, video)
    if next(analyzer.parameters()).is_cuda:
        torch.cuda.synchronize()
    return result, time.perf_counter() - start


def measure_clip(dataset, index, analyzers, codec, risk, policy, qps=QPS):
    start = time.perf_counter()
    source, _label, meta = dataset[index]
    rgb = (source.permute(1, 2, 3, 0).numpy() * 255).round().astype(np.uint8)
    decode_s = time.perf_counter() - start

    def identity_only():
        start = time.perf_counter()
        for qp in qps:
            codec._encode_decode_clip(rgb, qp=qp)
        return time.perf_counter() - start

    def full_selector():
        full_start = time.perf_counter()
        start = time.perf_counter()
        variants = make_candidates(rgb)
        variants_s = time.perf_counter() - start
        clean, clean_s = {}, 0.0
        for model in MODELS:
            (logits, feature), elapsed = measured_prediction(analyzers[model], source[None])
            clean[model] = (logits.softmax(1), feature)
            clean_s += elapsed
        points = []
        for qp in qps:
            anchor, candidates = {}, []
            candidate_costs = []
            for name in CANDIDATES:
                candidate = variants[name]
                _t, h, w, _ = candidate.shape
                start = time.perf_counter()
                reconstructed, native_bpp = codec._encode_decode_clip(candidate, qp=qp)
                codec_s = time.perf_counter() - start
                row = {"name": name, "bpp": normalized_bpp(native_bpp, h, w),
                       "signals": {}}
                model_s = 0.0
                for model in MODELS:
                    (logits, feature), elapsed = measured_prediction(
                        analyzers[model], as_video(reconstructed))
                    model_s += elapsed
                    if name == "identity128":
                        anchor[model] = (logits.softmax(1), feature)
                    row["signals"][model] = signals(logits, feature, clean[model],
                                                      anchor[model])
                candidates.append(row)
                candidate_costs.append({"name": name, "codec_s": codec_s,
                                        "both_analyzers_s": model_s})
            start = time.perf_counter()
            chosen = select(candidates, qp, policy, risk)
            selector_s = time.perf_counter() - start
            points.append({"qp": qp, "chosen": CANDIDATES[chosen],
                           "candidate_costs": candidate_costs, "selector_s": selector_s})
        return time.perf_counter() - full_start, variants_s, clean_s, points

    # Block by source clip; reverse the arm order for half of clips to reduce
    # systematic warm-up/drift bias. The seed is fixed and independent of results.
    reverse = int(hashlib.sha256(
        f"paper-runtime-order-20260924\0{meta['sequence_id']}".encode()
    ).hexdigest(), 16) % 2 == 1
    order = ("full", "identity") if reverse else ("identity", "full")
    elapsed = {}
    for arm in order:
        elapsed[arm] = full_selector() if arm == "full" else identity_only()
    full_s, variants_s, clean_s, points = elapsed["full"]
    baseline_s = elapsed["identity"]
    codec_s = sum(c["codec_s"] for point in points for c in point["candidate_costs"])
    analyzer_s = clean_s + sum(c["both_analyzers_s"] for point in points
                               for c in point["candidate_costs"])
    selector_s = sum(point["selector_s"] for point in points)
    return {"sequence_id": meta["sequence_id"], "decode_s": decode_s,
            "candidate_generation_s": variants_s, "clean_analyzers_s": clean_s,
            "arm_order": list(order), "qps": points,
            "candidate_codec_s": codec_s, "analyzer_s": analyzer_s,
            "selector_s": selector_s,
            "baseline_codec_calls": len(qps),
            "full_codec_calls": len(qps) * len(CANDIDATES),
            "baseline_analyzer_calls": 0,
            "full_analyzer_calls": len(MODELS) * (1 + len(qps) * len(CANDIDATES)),
            "identity_only_s": decode_s + baseline_s,
            "full_selector_s": decode_s + full_s,
            "overhead_ratio": (decode_s + full_s) / (decode_s + baseline_s)}


def summarize(rows):
    if not rows:
        raise ValueError("no timing records")
    operating_qps = [point["qp"] for point in rows[0]["qps"]]
    if any([point["qp"] for point in row["qps"]] != operating_qps for row in rows):
        raise ValueError("runtime records use different QPs")
    result = {"n": len(rows), "unit": "one 16-frame source clip",
              "qps": operating_qps,
              "baseline_codec_calls_per_clip": len(operating_qps),
              "full_codec_calls_per_clip": len(operating_qps) * len(CANDIDATES),
              "scope": "paired, order-balanced wall time of FFmpeg encode+decode; "
                       "codec-only baseline has no analyzer pass; full arm includes "
                       "candidate generation, all two-analyzer passes and selection; "
                       "model load/download excluded"}
    for key in ("identity_only_s", "full_selector_s", "overhead_ratio",
                "candidate_codec_s", "analyzer_s", "selector_s"):
        values = np.asarray([r[key] for r in rows], dtype=np.float64)
        result[key] = {"median": float(np.median(values)),
                       "mean": float(np.mean(values)),
                       "p95": float(np.percentile(values, 95))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--codec", choices=("h264", "h265"), required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--clips", type=int, default=20)
    parser.add_argument("--qps", default="40",
                        help="comma-separated QPs; use 40 for one deployment point or "
                             "30,35,40,45,50 for a full RD sweep")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not ffmpeg_available() or args.clips < 1:
        raise ValueError("ffmpeg/ffprobe and positive clip count required")
    try:
        qps = tuple(int(value) for value in args.qps.split(","))
    except ValueError as exc:
        raise ValueError("--qps must contain comma-separated integers") from exc
    if not qps or len(set(qps)) != len(qps) or any(qp not in QPS for qp in qps):
        raise ValueError(f"--qps must be a nonempty unique subset of {QPS}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    risk, frozen, policy = load_frozen(args.codec, config)
    dataset = VideoClipDataset(args.index, split=args.split,
                               num_frames=config["frames"],
                               frame_size=config["frame_size"],
                               temporal_stride=config["temporal_stride"],
                               train=False, return_metadata=True)
    indices = balanced_indices(dataset.samples, args.clips,
                               "paper-runtime-v1-20260924")
    if len(indices) != args.clips:
        raise ValueError("not enough source clips")
    manifest = {"experiment": "dual_v2_full_runtime", "codec": args.codec,
                "split": args.split, "clips": args.clips,
                "sample_ids": [clip_id(dataset.samples[i]) for i in indices],
                "code_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                "config_sha256": file_sha256(CONFIG),
                "index_sha256": file_sha256(args.index),
                "frozen_policy_sha256": digest(frozen), "risk_sha256": digest(risk),
                "policy": policy, "qps": list(qps), "candidates": list(CANDIDATES),
                "models": list(MODELS),
                "frames": config["frames"], "frame_size": config["frame_size"],
                "temporal_stride": config["temporal_stride"],
                "preset": config["preset"],
                "hardware": {"cpu": platform.processor(), "logical_cpus": os.cpu_count(),
                             "gpu": torch.cuda.get_device_name(0)
                             if torch.cuda.is_available() else None},
                "versions": {"python": platform.python_version(),
                             "torch": torch.__version__,
                             "torchvision": torchvision.__version__,
                             "ffmpeg": subprocess.check_output(
                                 ["ffmpeg", "-version"], text=True).splitlines()[0]},
                "comparison": "same clips/QPs; codec-only one encode+decode and "
                              "zero analyzer passes versus full six-candidate search"}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("output cache belongs to a different runtime run")
    write_json(manifest_path, manifest)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analyzers = {model: ActionRecognitionAnalyzer(model, clip_size=112).freeze().to(device)
                 for model in MODELS}
    codec = StandardCodec(args.codec, preset=config["preset"], strict_decode=True)
    rows = []
    for number, index in enumerate(indices, 1):
        path = args.out_dir / "cache" / f"clip_{number:04d}.json"
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
        else:
            row = measure_clip(dataset, index, analyzers, codec, risk, policy, qps)
            write_json(path, row)
        if row["sequence_id"] != clip_id(dataset.samples[index]):
            raise ValueError("stale runtime cache")
        rows.append(row)
        print(f"[runtime] {args.codec} {number}/{len(indices)}", flush=True)
    write_json(args.out_dir / "runtime_result.json",
               {"manifest": manifest, "summary": summarize(rows), "records": rows})
    print(json.dumps(summarize(rows), indent=2))


if __name__ == "__main__":
    main()
