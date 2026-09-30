#!/usr/bin/env python
"""Preregistered CAL-only H.265 motion-protected preprocessing screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from ops.codec_search_ar import QPS, as_video
from ops.dual_codec_search import digest, write_json
from ops.v7_pixel_proxy import find_planned_videos, sha256
from src.codecs.standard import StandardCodec, ffmpeg_available
from src.data.video_dataset import VideoClipDataset
from src.metrics.bd_rate import bd_metric, bd_rate
from src.models.codec_search import make_candidates, normalized_bpp
from src.tasks.action_recognition import (
    ActionRecognitionAnalyzer, _canon, kinetics_categories,
    kinetics_category_index,
)

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / "configs/v7_dev_proxy_plan.json"
PREREG = REPO / "docs/PREREGISTRATION_V8_MOTION_PILOT.md"
PREREG_COMMIT = "cb259971bd2dc6f201eeab14fa7d7af7a04a627e"
PLAN_SHA256 = "5b0a32d1d0a1759b50232f0c297ee35ee5cfe0cfdf00c53b587781baa5afebc2"
EXPERIMENT = "v8_motion_protect_h265_cal_pilot"
ARMS = ("identity128", "area112", "blur040_128", "motionprotect128")
MODELS = ("r2plus1d_18", "r3d_18", "mc3_18")
SHARDS = 4
N = 100
SEED = 20261005
DRAWS = 2000
COMPARISONS = (
    ("motion_vs_identity", "identity128", "motionprotect128"),
    ("motion_vs_area112", "area112", "motionprotect128"),
    ("motion_vs_uniform_blur", "blur040_128", "motionprotect128"),
    ("area112_vs_identity", "identity128", "area112"),
    ("uniform_blur_vs_identity", "identity128", "blur040_128"),
)


def git(*args: str) -> bytes:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={REPO.as_posix()}", *args], cwd=REPO
    )


def protocol() -> tuple[list[str], str, str]:
    git("merge-base", "--is-ancestor", PREREG_COMMIT, "HEAD")
    locked = git("show", f"{PREREG_COMMIT}:docs/PREREGISTRATION_V8_MOTION_PILOT.md")
    if PREREG.read_bytes().replace(b"\r\n", b"\n") != locked:
        raise ValueError("V8 pilot preregistration changed")
    if sha256(PLAN) != PLAN_SHA256:
        raise ValueError("source plan changed")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    ids = plan["stage_ids"]["calibration"][:N]
    if len(ids) != N or len(set(ids)) != N or len(plan["stage_ids"]["calibration"]) != 200:
        raise ValueError("invalid CAL source plan")
    head = git("rev-parse", "HEAD").decode().strip()
    for relative in ("ops/v8_motion_pilot.py", "kaggle/v8_motion_pilot_cell.sh"):
        git("ls-files", "--error-unmatch", relative)
        subprocess.run(
            ["git", "-c", f"safe.directory={REPO.as_posix()}",
             "diff", "--quiet", "HEAD", "--", relative],
            cwd=REPO, check=True, capture_output=True,
        )
    return ids, head, hashlib.sha256(locked).hexdigest()


def motion_protect(clip: np.ndarray) -> tuple[np.ndarray, float]:
    """Fixed pre-codec transform; return stream and protected-mask fraction."""
    if clip.shape != (16, 128, 128, 3) or clip.dtype != np.uint8:
        raise ValueError("expected uint8 RGB [16,128,128,3]")
    gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in clip])
    delta = np.abs(np.diff(gray.astype(np.int16), axis=0)).astype(np.float32)
    motion = np.zeros_like(gray, dtype=np.float32)
    motion[:-1] = delta
    motion[1:] = np.maximum(motion[1:], delta)
    threshold = float(np.quantile(motion, 0.8))
    raw_mask = (motion > threshold).astype(np.float32)
    kernel = np.ones((5, 5), dtype=np.uint8)
    dilated = np.stack([cv2.dilate(frame, kernel) for frame in raw_mask])
    mask = np.stack([cv2.GaussianBlur(frame, (0, 0), 1.0) for frame in dilated])
    mask = np.clip(mask, 0.0, 1.0)
    blurred = np.stack([cv2.GaussianBlur(frame, (0, 0), 1.5) for frame in clip])
    out = np.clip(np.rint(mask[..., None] * clip.astype(np.float32)
                          + (1.0 - mask[..., None]) * blurred.astype(np.float32)),
                  0, 255).astype(np.uint8)
    return out, float(dilated.mean())


def decode_source(path: Path) -> np.ndarray:
    dataset = VideoClipDataset.__new__(VideoClipDataset)
    dataset.num_frames, dataset.frame_size = 16, 128
    dataset.temporal_stride, dataset.train = 2, False
    return dataset._read_clip(str(path))


def validate_record(row: dict, key: str, source_sha: str, label: int) -> None:
    if (row.get("sequence_id") != key or row.get("source_sha256") != source_sha
            or not isinstance(source_sha, str) or len(source_sha) != 64
            or any(char not in "0123456789abcdef" for char in source_sha)
            or row.get("label") != label or len(row.get("measurements", [])) != len(QPS)):
        raise ValueError("stale or incomplete V8 source record")
    fraction = row.get("protected_mask_fraction")
    if not isinstance(fraction, (int, float)) or not 0 <= fraction <= 1:
        raise ValueError("invalid motion mask fraction")
    for measurement, qp in zip(row["measurements"], QPS, strict=True):
        if measurement.get("qp") != qp or set(measurement.get("arms", {})) != set(ARMS):
            raise ValueError("V8 QP/arm record incomplete")
        for arm, value in measurement["arms"].items():
            if (value.get("name") != arm or value.get("coded_bytes", 0) <= 0
                    or value.get("bpp", 0) <= 0
                    or set(value.get("analyzers", {})) != set(MODELS)):
                raise ValueError("invalid coded V8 arm")
            if abs(value["bpp"] - 8 * value["coded_bytes"] / (16 * 128 * 128)) > 1e-9:
                raise ValueError("V8 bpp must use source-pixel denominator")
            for model in MODELS:
                pred = value["analyzers"][model]
                if (not 0 <= pred["predicted_class_index"] < 400
                        or pred["correct"] != (pred["predicted_class_index"] == label)):
                    raise ValueError("inconsistent analyzer outcome")


def evaluate_source(key: str, path: Path, label: int, analyzers: dict,
                    codec: StandardCodec) -> dict:
    cap = cv2.VideoCapture(str(path))
    ok, _ = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"undecodable CAL source: {key}")
    source_sha = sha256(path)
    source = decode_source(path)
    variants = make_candidates(source)
    motion, fraction = motion_protect(source)
    variants["motionprotect128"] = motion
    import torch
    measurements = []
    for qp in QPS:
        values = {}
        for arm in ARMS:
            candidate = variants[arm]
            _, h, w, _ = candidate.shape
            start = time.perf_counter()
            reconstructed, native_bpp = codec._encode_decode_clip(candidate, qp=qp)
            encode_s = time.perf_counter() - start
            coded_bytes = int(round(native_bpp * 16 * h * w / 8))
            scores = {}
            for model, analyzer in analyzers.items():
                if next(analyzer.parameters()).is_cuda:
                    torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.no_grad():
                    video = as_video(reconstructed).to(next(analyzer.parameters()).device)
                    prediction = int(analyzer.predict(video).argmax(1).item())
                if next(analyzer.parameters()).is_cuda:
                    torch.cuda.synchronize()
                scores[model] = {
                    "predicted_class_index": prediction, "correct": prediction == label,
                    "inference_s": time.perf_counter() - start,
                }
            values[arm] = {
                "name": arm, "coded_bytes": coded_bytes,
                "bpp": normalized_bpp(native_bpp, h, w),
                "encode_decode_s": encode_s, "analyzers": scores,
            }
        measurements.append({"qp": qp, "arms": values})
    row = {"sequence_id": key, "source_sha256": source_sha, "label": label,
           "protected_mask_fraction": fraction, "measurements": measurements}
    validate_record(row, key, source_sha, label)
    return row


def run_shard(source_root: Path, shard: int, out_dir: Path) -> dict:
    if shard not in range(SHARDS) or not source_root.is_dir() or not ffmpeg_available():
        raise ValueError("invalid source root/shard or missing ffmpeg")
    ids, head, prereg_sha = protocol()
    assigned = ids[shard::SHARDS]
    paths = find_planned_videos(source_root, assigned)
    if any(kinetics_categories(model) != kinetics_categories(MODELS[0]) for model in MODELS):
        raise ValueError("Kinetics class order differs between analyzers")
    categories = kinetics_category_index(MODELS[0])
    import torch
    import torchvision
    torch.manual_seed(53)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analyzers = {model: ActionRecognitionAnalyzer(model, clip_size=112).freeze().to(device)
                 for model in MODELS}
    codec = StandardCodec("h265", preset="medium", strict_decode=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, key in enumerate(assigned, 1):
        label = categories.get(_canon(key.split("/")[0]))
        if label is None:
            raise ValueError(f"unmapped Kinetics category: {key}")
        checkpoint = out_dir / "cache" / f"clip_{number:03d}.json"
        source_sha = sha256(paths[key])
        if checkpoint.exists():
            row = json.loads(checkpoint.read_text(encoding="utf-8"))
        else:
            row = evaluate_source(key, paths[key], label, analyzers, codec)
            write_json(checkpoint, row)
        validate_record(row, key, source_sha, label)
        rows.append(row)
        print(f"[V8 CAL] shard={shard} {number}/{len(assigned)}", flush=True)
    raw = out_dir / "shard_records.jsonl"
    raw.write_bytes("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                            for row in rows).encode("utf-8"))
    manifest = {
        "experiment": EXPERIMENT, "scope": "reused CAL; development only",
        "codec": "h265", "shard": shard, "shards": SHARDS, "n": len(rows),
        "source_ids": assigned, "source_fingerprint": digest(ids),
        "source_plan_sha256": PLAN_SHA256, "preregistration_commit": PREREG_COMMIT,
        "preregistration_sha256": prereg_sha, "code_commit": head,
        "records_sha256": sha256(raw), "qps": list(QPS), "arms": list(ARMS),
        "analyzers": list(MODELS), "bootstrap_seed": SEED, "bootstrap_draws": DRAWS,
        "bootstrap_unit": "source video; all QPs, arms and analyzers paired",
        "versions": {
            "python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "opencv": cv2.__version__,
            "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
        },
        "device": str(device),
    }
    write_json(out_dir / "manifest.json", manifest)
    return {"shard": shard, "n": len(rows), "records_sha256": sha256(raw)}


def curves(rows: list[dict], arm: str, model: str) -> dict:
    return {
        str(qp): {
            "n": len(rows),
            "bpp": float(np.mean([row["measurements"][i]["arms"][arm]["bpp"] for row in rows])),
            "top1": float(np.mean([
                row["measurements"][i]["arms"][arm]["analyzers"][model]["correct"]
                for row in rows])),
        }
        for i, qp in enumerate(QPS)
    }


def compare(a: dict, b: dict) -> dict:
    anchor, trial = [a[str(qp)] for qp in QPS], [b[str(qp)] for qp in QPS]
    ra, aa = [p["bpp"] for p in anchor], [p["top1"] for p in anchor]
    rb, ab = [p["bpp"] for p in trial], [p["top1"] for p in trial]
    rate = bd_rate(ra, aa, rb, ab)
    accuracy = 100 * bd_metric(ra, aa, rb, ab)
    return {
        "bd_rate_top1_pct": rate if math.isfinite(rate) else None,
        "bd_accuracy_top1_pp": accuracy if math.isfinite(accuracy) else None,
        "min_same_qp_top1_gap_pp": 100 * min(x - y for x, y in zip(ab, aa)),
    }


def summarize(rows: list[dict], anchor: str, trial: str, model: str,
              resamples: np.ndarray) -> dict:
    anchor_curve, trial_curve = curves(rows, anchor, model), curves(rows, trial, model)
    point = compare(anchor_curve, trial_curve)
    samples = {key: [] for key in ("bd_rate_top1_pct", "bd_accuracy_top1_pp")}
    for indices in resamples:
        sampled = [rows[i] for i in indices]
        values = compare(curves(sampled, anchor, model), curves(sampled, trial, model))
        for key in samples:
            if values[key] is not None and np.isfinite(values[key]):
                samples[key].append(values[key])
    intervals = {
        key: {"valid_draws": len(values), "requested_draws": len(resamples),
              "ci95": np.percentile(values, [2.5, 97.5]).tolist() if values else None}
        for key, values in samples.items()
    }
    return {"anchor_arm": anchor, "trial_arm": trial, "anchor_curve": anchor_curve,
            "trial_curve": trial_curve, "metrics": point, "bootstrap": intervals}


def merge(shard_dirs: list[Path], out: Path) -> dict:
    if len(shard_dirs) != SHARDS or out.exists():
        raise ValueError("four shard directories and a fresh output required")
    ids, head, prereg_sha = protocol()
    bundles = {}
    categories = kinetics_category_index(MODELS[0])
    for folder in shard_dirs:
        manifest_path, raw = folder / "manifest.json", folder / "shard_records.jsonl"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shard = manifest.get("shard")
        if shard in bundles or shard not in range(SHARDS):
            raise ValueError("duplicate or invalid V8 shard")
        expected = ids[shard::SHARDS]
        rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]
        if (manifest.get("experiment") != EXPERIMENT
                or manifest.get("source_ids") != expected
                or manifest.get("source_fingerprint") != digest(ids)
                or manifest.get("source_plan_sha256") != PLAN_SHA256
                or manifest.get("preregistration_commit") != PREREG_COMMIT
                or manifest.get("preregistration_sha256") != prereg_sha
                or manifest.get("code_commit") != head
                or manifest.get("records_sha256") != sha256(raw)
                or manifest.get("qps") != list(QPS)
                or manifest.get("arms") != list(ARMS)
                or manifest.get("analyzers") != list(MODELS)
                or manifest.get("bootstrap_seed") != SEED
                or manifest.get("bootstrap_draws") != DRAWS
                or len(rows) != len(expected)):
            raise ValueError("mismatched or incomplete V8 shard")
        for row, key in zip(rows, expected, strict=True):
            validate_record(row, key, row.get("source_sha256"), categories[_canon(key.split("/")[0])])
        bundles[shard] = (manifest, rows, sha256(raw))
    if set(bundles) != set(range(SHARDS)):
        raise ValueError("missing V8 shard")
    by_id = {row["sequence_id"]: row for _, rows, _ in bundles.values() for row in rows}
    if len(by_id) != N or set(by_id) != set(ids):
        raise ValueError("V8 shards overlap or omit CAL sources")
    ordered = [by_id[key] for key in ids]
    rng = np.random.default_rng(SEED)
    resamples = rng.integers(0, N, size=(DRAWS, N))
    comparisons = {
        name: {model: summarize(ordered, anchor, trial, model, resamples)
               for model in MODELS}
        for name, anchor, trial in COMPARISONS
    }
    main = comparisons["motion_vs_identity"]
    go = all(
        main[model]["metrics"]["bd_rate_top1_pct"] is not None
        and main[model]["metrics"]["bd_rate_top1_pct"] < 0
        and main[model]["metrics"]["min_same_qp_top1_gap_pp"] >= -1.0
        for model in MODELS
    )
    report = {
        "experiment": EXPERIMENT, "scope": "reused CAL developmental pilot; no holdout",
        "codec": "h265", "n": N, "source_ids": ids,
        "source_fingerprint": digest(ids), "source_plan_sha256": PLAN_SHA256,
        "preregistration_commit": PREREG_COMMIT, "preregistration_sha256": prereg_sha,
        "analysis_code_commit": head, "bootstrap_seed": SEED,
        "bootstrap_draws": DRAWS,
        "bootstrap_unit": "source video; all QPs, arms and analyzers paired",
        "shard_manifests": [bundles[i][0] for i in range(SHARDS)],
        "shard_records_sha256": [bundles[i][2] for i in range(SHARDS)],
        "source_video_sha256": {key: by_id[key]["source_sha256"] for key in ids},
        "protected_mask_fraction": {
            "mean": float(np.mean([row["protected_mask_fraction"] for row in ordered])),
            "median": float(np.median([row["protected_mask_fraction"] for row in ordered])),
        },
        "comparisons": comparisons, "go_no_go": go,
        "primary_technical_target": all(
            main[model]["metrics"]["bd_rate_top1_pct"] is not None
            and main[model]["metrics"]["bd_rate_top1_pct"] < -10 for model in MODELS[:2]
        ),
        "original_confirmatory_gate": "NOT ASSESSED: CAL-only reused sources",
    }
    write_json(out, report)
    result_sha = sha256(out)
    out.with_suffix(".sha256").write_text(f"{result_sha}  {out.name}\n", encoding="ascii")
    return {"result_sha256": result_sha, "go_no_go": go}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    shard = subs.add_parser("shard")
    shard.add_argument("--source-root", type=Path, required=True)
    shard.add_argument("--shard", type=int, choices=range(SHARDS), required=True)
    shard.add_argument("--out-dir", type=Path, required=True)
    merged = subs.add_parser("merge")
    merged.add_argument("--shard-dir", type=Path, action="append", required=True)
    merged.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = (run_shard(args.source_root, args.shard, args.out_dir)
              if args.command == "shard" else merge(args.shard_dir, args.out))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
