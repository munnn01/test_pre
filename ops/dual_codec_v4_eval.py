#!/usr/bin/env python
"""Development-only 200-VAL-clip comparison of frozen V2 versus frozen V4.

V4 must already be fitted/frozen from TRAIN fit/calibration. The old VAL set
was used during V2/V3 development: this is NOT a new holdout. All three
analyzers share one selected bitstream per codec/QP. mc3 outcomes are read
only after both selectors have made their choices.
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

from ops.codec_search_ar import QPS
from ops.dual_codec_lowqp_v3 import load_pilot
from ops.dual_codec_search import digest, write_json
from ops.dual_codec_v4_fit import ANALYZERS, OUTCOME_FIELDS, select_v4
from ops.paper_heldout_mc3 import compare, timed_inference
from ops.rcts_pilot import clip_id
from ops.v4_mc3_fit_cache import source_motion_features
from src.codecs.standard import StandardCodec, ffmpeg_available
from src.data.video_dataset import VideoClipDataset
from src.models.codec_search import make_candidates, normalized_bpp
from src.models.dual_codec_search import observations, risk_scores, select_observations
from src.tasks.action_recognition import ActionRecognitionAnalyzer, kinetics_categories

REPO = Path(__file__).resolve().parents[1]
ARMS = ("v2_frozen", "v4_frozen")
MODEL = "mc3_18"


def load_v4(directory: Path, codec: str, pilot_manifest: dict,
            v2_frozen: dict) -> tuple[dict, dict]:
    model = json.loads((directory / "risk_model.json").read_text(encoding="utf-8"))
    frozen = json.loads((directory / "frozen_policy.json").read_text(encoding="utf-8"))
    calibration = json.loads((directory / "calibration.json").read_text(encoding="utf-8"))
    if (frozen.get("schema") != 1 or frozen.get("codec") != codec
            or frozen.get("risk_model_sha256") != digest(model)
            or frozen.get("calibration_sha256") != digest(calibration)
            or frozen.get("v2_manifest_sha256") != digest(pilot_manifest)
            or frozen.get("v2_frozen_policy_sha256") != digest(v2_frozen)
            or frozen.get("selected_policy") != calibration.get("selected_policy")):
        raise ValueError("V4 frozen-policy provenance differs")
    return model, frozen


def evaluate_clip(dataset: VideoClipDataset, index: int, pilot: dict,
                  v2_risk: dict, v2_policy: dict, v4_model: dict,
                  v4_policy: dict, analyzer: ActionRecognitionAnalyzer,
                  codec: StandardCodec) -> dict:
    source, label, meta = dataset[index]
    if meta["sequence_id"] != pilot["sequence_id"]:
        raise ValueError("pilot and decoded video IDs differ")
    capture = cv2.VideoCapture(dataset.samples[index]["path"])
    ok, _ = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"undecodable source: {meta['sequence_id']}")
    rgb = (source.permute(1, 2, 3, 0).numpy() * 255).round().astype(np.uint8)
    variants = make_candidates(rgb)
    source_features = source_motion_features(rgb)
    measurements = []
    for cached in pilot["measurements"]:
        qp = cached["qp"]
        obs = observations(cached["candidates"])
        v2_index = select_observations(obs, qp, v2_policy,
                                       risk_scores(obs, qp, v2_risk))
        v4_index = select_v4(obs, qp, source_features, v4_model,
                             v4_policy, v2_risk, v2_policy)
        choices = {"v2_frozen": obs[v2_index]["name"],
                   "v4_frozen": obs[v4_index]["name"]}
        names = dict.fromkeys(("identity128", *choices.values()))
        streams = {}
        by_name = {r["name"]: r for r in cached["candidates"]}
        for name in names:
            candidate = variants[name]
            _t, h, w, _ = candidate.shape
            start = time.perf_counter()
            reconstructed, native = codec._encode_decode_clip(candidate, qp=qp)
            encoding = time.perf_counter() - start
            prediction, inference = timed_inference(analyzer, reconstructed)
            bpp = normalized_bpp(native, h, w)
            if abs(bpp - by_name[name]["bpp"]) > 1e-9:
                raise ValueError(f"bitstream differs from V2 pilot: {name} QP{qp}")
            streams[name] = {"bpp": bpp,
                "correct": {"r2plus1d_18": by_name[name]["correct"],
                            "r3d_18": by_name[name]["cross_correct"],
                            "mc3_18": bool(prediction == label)},
                "encode_decode_s": encoding, "mc3_inference_s": inference}
        measurements.append({"qp": qp, "choices": choices, "streams": streams})
    return {"sequence_id": meta["sequence_id"], "measurements": measurements}


def _curve(records: list[dict], arm: str, model: str) -> dict:
    curve = {}
    for position, qp in enumerate(QPS):
        points = []
        for record in records:
            measurement = record["measurements"][position]
            if measurement["qp"] != qp:
                raise ValueError("QP order differs")
            name = "identity128" if arm == "anchor" else measurement["choices"][arm]
            points.append(measurement["streams"][name])
        curve[str(qp)] = {"n": len(points),
                          "bpp": float(np.mean([p["bpp"] for p in points])),
                          "top1": float(np.mean([p["correct"][model] for p in points]))}
    return curve


def summarize(records: list[dict], draws: int) -> dict:
    if not records or draws < 1:
        raise ValueError("nonempty records and positive bootstrap required")
    rng = np.random.default_rng(20260926)
    samples = rng.integers(0, len(records), size=(draws, len(records)))
    results = {}
    for arm in ARMS:
        results[arm] = {}
        for model in ANALYZERS:
            anchor = _curve(records, "anchor", model)
            trial = _curve(records, arm, model)
            point = compare(anchor, trial)
            distributions = {key: [] for key in ("bd_rate_top1_pct", "bd_accuracy_top1_pp")}
            for picked in samples:
                selected = [records[i] for i in picked]
                metrics = compare(_curve(selected, "anchor", model),
                                  _curve(selected, arm, model))
                for key in distributions:
                    value = metrics[key]
                    if value is not None and np.isfinite(value):
                        distributions[key].append(value)
            choices = {name: sum(m["choices"][arm] == name
                                 for record in records for m in record["measurements"])
                       for name in {n for record in records for m in record["measurements"]
                                    for n in m["choices"].values()}}
            results[arm][model] = {"anchor_curve": anchor, "trial_curve": trial,
                "metrics": point, "choices": choices,
                "bootstrap": {key: {"requested_draws": draws,
                                    "valid_draws": len(values),
                                    "ci95": np.percentile(values, [2.5, 97.5]).tolist()
                                    if values else None}
                              for key, values in distributions.items()}}
    return {"n_videos": len(records), "arms": results,
            "bootstrap_unit": "whole video; all five QPs paired",
            "scope": "Previously inspected V2/V3 VAL; development comparison only",
            "timing_scope": "Only distinct chosen streams were re-encoded; full six-candidate search cost is not measured here."}


def evaluate(args: argparse.Namespace) -> None:
    if not ffmpeg_available():
        raise ValueError("ffmpeg/ffprobe required")
    if kinetics_categories(MODEL) != kinetics_categories("r3d_18"):
        raise ValueError("mc3 category mapping differs")
    pilot_manifest, v2_risk, v2_frozen, pilot = load_pilot(
        args.archive, args.codec, stages=("dev",))
    model, v4_frozen = load_v4(args.v4_dir, args.codec,
                               pilot_manifest, v2_frozen)
    expected_ids = pilot_manifest["split_ids"]["dev"]
    dataset = VideoClipDataset(args.index, split="val", num_frames=16,
                               frame_size=128, temporal_stride=2,
                               train=False, return_metadata=True)
    locations = {clip_id(row): i for i, row in enumerate(dataset.samples)}
    if (len(locations) != len(dataset.samples) or len(expected_ids) != 200
            or any(key not in locations for key in expected_ids)):
        raise ValueError("incomplete V2 DEV source cache")
    positions = list(range(args.shard, 200, 2))
    keys = [expected_ids[i] for i in positions]
    manifest = {"experiment": "dual_codec_v4_val_development", "codec": args.codec,
        "shard": args.shard, "shards": 2, "sample_ids": keys,
        "sample_fingerprint": digest(keys), "dev_fingerprint": digest(expected_ids),
        "source_archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "pilot_manifest_sha256": digest(pilot_manifest),
        "v2_risk_sha256": digest(v2_risk), "v2_frozen_sha256": digest(v2_frozen),
        "v4_frozen_sha256": digest(v4_frozen), "v4_model_sha256": digest(model),
        "index_sha256": hashlib.sha256(args.index.read_bytes()).hexdigest(),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
             cwd=REPO, text=True).strip(), "qps": list(QPS),
        "versions": {"python": platform.python_version(),
                     "torch": torch.__version__, "torchvision": torchvision.__version__},
        "scope": "Previously inspected VAL; not a source-disjoint holdout"}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("stale V4 VAL output directory")
    write_json(manifest_path, manifest)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analyzer = ActionRecognitionAnalyzer(MODEL, clip_size=112).freeze().to(device)
    codec = StandardCodec(args.codec, preset="medium", strict_decode=True)
    rows = []
    for number, position in enumerate(positions, 1):
        key = expected_ids[position]
        path = args.out_dir / "cache" / f"clip_{number:04d}.json"
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
        else:
            row = evaluate_clip(dataset, locations[key], pilot["dev"][position],
                                v2_risk, v2_frozen["selected_policy"], model,
                                v4_frozen["selected_policy"], analyzer, codec)
            write_json(path, row)
        if row["sequence_id"] != key or [m["qp"] for m in row["measurements"]] != list(QPS):
            raise ValueError("incomplete V4 VAL per-video cache")
        rows.append(row)
        print(f"[v4-val] {args.codec} shard={args.shard} {number}/100", flush=True)
    out = args.out_dir / "shard_records.jsonl"
    temporary = out.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    temporary.replace(out)
    write_json(args.out_dir / "shard_result.json", {"manifest": manifest,
        "diagnostic_only": summarize(rows, args.bootstrap),
        "warning": "Merge 2 shards; VAL is not new holdout."})


def merge(args: argparse.Namespace) -> None:
    if len(args.shard_dir) != 2:
        raise ValueError("exactly two VAL shards required")
    bundles = []
    for directory in args.shard_dir:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in
                (directory / "shard_records.jsonl").read_text(encoding="utf-8").splitlines()]
        if (manifest.get("experiment") != "dual_codec_v4_val_development"
                or manifest.get("codec") != args.codec or len(rows) != 100
                or [row["sequence_id"] for row in rows] != manifest["sample_ids"]
                or digest(manifest["sample_ids"]) != manifest["sample_fingerprint"]):
            raise ValueError("invalid V4 VAL shard")
        bundles.append((manifest, rows))
    bundles.sort(key=lambda item: item[0]["shard"])
    if [item[0]["shard"] for item in bundles] != [0, 1]:
        raise ValueError("missing or duplicate V4 VAL shard")
    common = ("experiment", "codec", "dev_fingerprint", "source_archive_sha256",
              "pilot_manifest_sha256", "v2_risk_sha256", "v2_frozen_sha256",
              "v4_frozen_sha256", "v4_model_sha256", "index_sha256", "code_commit", "qps")
    if any(bundles[0][0][key] != bundles[1][0][key] for key in common):
        raise ValueError("V4 VAL shard provenance differs")
    ids = [None] * 200
    by_id = {}
    for manifest, rows in bundles:
        ids[manifest["shard"]::2] = manifest["sample_ids"]
        by_id.update({row["sequence_id"]: row for row in rows})
    if len(by_id) != 200 or digest(ids) != bundles[0][0]["dev_fingerprint"]:
        raise ValueError("V4 VAL merged fingerprint differs")
    report = {"experiment": "dual_codec_v4_val_development", "codec": args.codec,
              "source_manifests": [item[0] for item in bundles],
              "result": summarize([by_id[key] for key in ids], args.bootstrap)}
    write_json(args.out, report)
    print(json.dumps({"codec": args.codec, "n": 200,
                      "metrics": {arm: {model: value["metrics"] for model, value in by_model.items()}
                                  for arm, by_model in report["result"]["arms"].items()}},
                     indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluator = sub.add_parser("evaluate")
    evaluator.add_argument("--archive", type=Path, required=True)
    evaluator.add_argument("--v4-dir", type=Path, required=True)
    evaluator.add_argument("--index", type=Path, required=True)
    evaluator.add_argument("--codec", choices=("h264", "h265"), required=True)
    evaluator.add_argument("--shard", type=int, choices=(0, 1), required=True)
    evaluator.add_argument("--out-dir", type=Path, required=True)
    evaluator.add_argument("--bootstrap", type=int, default=100)
    merger = sub.add_parser("merge")
    merger.add_argument("--codec", choices=("h264", "h265"), required=True)
    merger.add_argument("--shard-dir", type=Path, action="append", required=True)
    merger.add_argument("--out", type=Path, required=True)
    merger.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    (evaluate if args.command == "evaluate" else merge)(args)


if __name__ == "__main__":
    main()
