#!/usr/bin/env python
"""Fit a shared V2 decoder restorer using two frozen analyzers; never load MC3."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from ops.codec_search_ar import as_video
from ops.dual_codec_lowqp_v3 import load_pilot
from ops.dual_codec_search import digest, write_json
from ops.rcts_pilot import clip_id
from src.codecs.standard import StandardCodec, ffmpeg_available
from src.data.video_dataset import VideoClipDataset
from src.models.codec_search import make_candidates, normalized_bpp
from src.models.dual_codec_search import MODELS, select
from src.models.v8_restorer import V8Restorer
from src.tasks.action_recognition import ActionRecognitionAnalyzer

REPO = Path(__file__).resolve().parents[1]
FIT_QPS = (30, 35, 40, 45, 50)
CAL_QPS = (30, 35, 40)
EPOCH_MODES = ("pixel", "pixel", "semantic")
SEED = 20260927


def fixed_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(2)


def data_lookup(index: Path, ids: list[str]) -> tuple[VideoClipDataset, dict[str, int]]:
    dataset = VideoClipDataset(index, split="train", num_frames=16, frame_size=128,
                               temporal_stride=2, train=False, return_metadata=True)
    locations = {clip_id(row): i for i, row in enumerate(dataset.samples)}
    if len(locations) != len(dataset.samples) or any(key not in locations for key in ids):
        raise ValueError("frozen TRAIN clip identities absent or duplicated")
    return dataset, locations


def encoded_point(dataset: VideoClipDataset, index: int, record: dict, qp: int,
                  codec: StandardCodec, policy: dict, risk: dict
                  ) -> tuple[torch.Tensor, torch.Tensor, int, float, int]:
    clean, label, meta = dataset[index]
    if meta["sequence_id"] != record["sequence_id"]:
        raise ValueError("cached source ID differs from clip")
    source = (clean.permute(1, 2, 3, 0).numpy() * 255).round().astype(np.uint8)
    measurement = next(m for m in record["measurements"] if m["qp"] == qp)
    choice = measurement["candidates"][select(measurement["candidates"], qp, policy, risk)]
    candidate = make_candidates(source)[choice["name"]]
    native_size = int(candidate.shape[1])
    reconstruction, native_bpp = codec._encode_decode_clip(candidate, qp=qp)
    bpp = normalized_bpp(native_bpp, native_size, native_size)
    if abs(bpp - float(choice["bpp"])) > 1e-9:
        raise ValueError("V2 selected bitstream differs from frozen pilot cache")
    return clean[None], as_video(reconstruction), native_size, bpp, label


def pixel_loss(restored: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    value = F.l1_loss(restored, clean)
    temporal = F.l1_loss(restored[:, :, 1:] - restored[:, :, :-1],
                         clean[:, :, 1:] - clean[:, :, :-1])
    return value + .25 * temporal


def semantic_loss(restored: torch.Tensor, clean: torch.Tensor,
                  teachers: dict[str, ActionRecognitionAnalyzer]) -> torch.Tensor:
    parts = []
    for model in MODELS:
        teacher = teachers[model]
        with torch.no_grad():
            target = teacher.features(clean)[2]
        actual = teacher.features(restored)[2]
        parts.append((actual - target).abs().mean() /
                     target.detach().abs().mean().clamp_min(.05))
    return torch.stack(parts).mean()


def checkpoint(out: Path, name: str, model: V8Restorer, *, epoch: int,
               metadata: dict) -> None:
    path = out / f"{name}.pth"
    temporary = path.with_suffix(".tmp")
    torch.save({"schema": 1, "model": model.state_dict(), "epoch": epoch,
                "metadata": metadata}, temporary)
    temporary.replace(path)


def train(args: argparse.Namespace) -> None:
    if not ffmpeg_available():
        raise ValueError("ffmpeg/ffprobe required")
    fixed_seeds()
    pilot_manifest, risk, frozen, records = load_pilot(
        args.archive, args.codec, stages=("fit", "calibration"))
    ids = pilot_manifest["split_ids"]["fit"] + pilot_manifest["split_ids"]["calibration"]
    dataset, locations = data_lookup(args.index, ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise ValueError("GPU required for V8 teacher-feature training")
    policy = frozen["selected_policy"]
    manifest = {"experiment": "v8_shared_decoder_h264", "codec": args.codec,
        "fit_ids": pilot_manifest["split_ids"]["fit"],
        "cal_ids": pilot_manifest["split_ids"]["calibration"],
        "fit_fingerprint": digest(pilot_manifest["split_ids"]["fit"]),
        "cal_fingerprint": digest(pilot_manifest["split_ids"]["calibration"]),
        "index_sha256": hashlib.sha256(args.index.read_bytes()).hexdigest(),
        "pilot_archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "frozen_policy_sha256": digest(frozen), "risk_sha256": digest(risk),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
             cwd=REPO, text=True).strip(),
        "qps_fit": list(FIT_QPS), "qps_cal": list(CAL_QPS),
        "epochs": list(EPOCH_MODES), "seed": SEED,
        "teachers": list(MODELS), "excluded_analyzer": "mc3_18",
        "torch": torch.__version__, "python": platform.python_version()}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("stale V8 training directory")
    write_json(manifest_path, manifest)
    model = V8Restorer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-5)
    codec = StandardCodec(args.codec, preset="medium", strict_decode=True)
    teachers = None
    train_rows = records["fit"]
    for epoch, mode in enumerate(EPOCH_MODES):
        if mode == "semantic" and teachers is None:
            teachers = {name: ActionRecognitionAnalyzer(name, clip_size=112).freeze().to(device)
                        for name in MODELS}
        order = np.random.default_rng(SEED + epoch).permutation(len(train_rows))
        total = 0.
        model.train()
        for step, position in enumerate(order, 1):
            position = int(position)
            record = train_rows[position]
            qp = FIT_QPS[(position + epoch) % len(FIT_QPS)]
            clean, decoded, native, _rate, _label = encoded_point(
                dataset, locations[record["sequence_id"]], record, qp, codec, policy, risk)
            clean, decoded = clean.to(device), decoded.to(device)
            restored = model(decoded, qp, native)
            loss = pixel_loss(restored, clean)
            if mode == "semantic":
                loss = loss + .015 * semantic_loss(restored, clean, teachers)
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach())
            if step % 25 == 0 or step == len(order):
                print(f"[fit] epoch={epoch} mode={mode} {step}/{len(order)} "
                      f"mean_loss={total/step:.6f}", flush=True)
        name = "pixel" if mode == "pixel" else "semantic"
        checkpoint(args.out_dir, name, model, epoch=epoch,
                   metadata={"device": str(device), "manifest_sha256": digest(manifest)})
        write_json(args.out_dir / f"epoch_{epoch}.json",
                   {"epoch": epoch, "mode": mode, "mean_loss": total / len(order)})
    del teachers
    torch.cuda.empty_cache()
    calibrate(args, dataset, locations, records["calibration"], codec, policy, risk,
              manifest, device)


@torch.no_grad()
def calibrate(args: argparse.Namespace, dataset: VideoClipDataset,
              locations: dict[str, int], rows: list[dict], codec: StandardCodec,
              policy: dict, risk: dict, manifest: dict, device: torch.device) -> None:
    candidates = {}
    for name in ("pixel", "semantic"):
        artifact = torch.load(args.out_dir / f"{name}.pth", map_location="cpu",
                              weights_only=True)
        if artifact["metadata"]["manifest_sha256"] != digest(manifest):
            raise ValueError("checkpoint manifest mismatch")
        model = V8Restorer().to(device).eval()
        model.load_state_dict(artifact["model"], strict=True)
        candidates[name] = model
    teachers = {name: ActionRecognitionAnalyzer(name, clip_size=112).freeze().to(device)
                for name in MODELS}
    table = {name: {str(q): {model: {"correct": 0, "ce_sum": 0.}
                                for model in MODELS} for q in CAL_QPS}
             for name in ("v2", "pixel", "semantic")}
    for number, record in enumerate(rows, 1):
        for qp in CAL_QPS:
            clean, decoded, native, _rate, label = encoded_point(
                dataset, locations[record["sequence_id"]], record, qp, codec, policy, risk)
            decoded = decoded.to(device)
            videos = {"v2": decoded,
                      **{name: net(decoded, qp, native) for name, net in candidates.items()}}
            for name, video in videos.items():
                for teacher_name, teacher in teachers.items():
                    logits = teacher.predict(video)
                    cell = table[name][str(qp)][teacher_name]
                    cell["correct"] += int(logits.argmax(1).item() == label)
                    cell["ce_sum"] += float(F.cross_entropy(
                        logits, torch.tensor([label], device=device)))
        if number % 25 == 0 or number == len(rows):
            print(f"[cal] {number}/{len(rows)}", flush=True)
    criteria = {}
    baseline = table["v2"]
    for name in ("pixel", "semantic"):
        gaps = [table[name][str(q)][model]["correct"] - baseline[str(q)][model]["correct"]
                for q in CAL_QPS for model in MODELS]
        mean_ce = np.mean([table[name][str(q)][model]["ce_sum"] / len(rows)
                           for q in CAL_QPS for model in MODELS])
        criteria[name] = {"min_correct_gap_count": min(gaps),
                          "correct_gaps": gaps, "mean_ce": float(mean_ce),
                          "eligible": bool(min(gaps) >= -2)}
    baseline_ce = np.mean([baseline[str(q)][model]["ce_sum"] / len(rows)
                           for q in CAL_QPS for model in MODELS])
    eligible = [name for name in criteria if criteria[name]["eligible"]
                and criteria[name]["mean_ce"] < baseline_ce]
    winner = min(eligible, key=lambda name: criteria[name]["mean_ce"]) if eligible else "v2"
    report = {"experiment": manifest["experiment"], "codec": args.codec,
        "manifest_sha256": digest(manifest), "n_cal": len(rows),
        "qps": list(CAL_QPS), "table": table,
        "criteria": criteria, "baseline_mean_ce": float(baseline_ce),
        "selected": winner,
        "selection_rule": "Two known analyzers only: every QP/model <=2/200 extra errors; "
                          "choose lowest mean CE below V2; otherwise V2 fallback.",
        "mc3_access": False}
    write_json(args.out_dir / "calibration.json", report)
    print(f"[selected] {winner} criteria={criteria}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--codec", choices=("h264",), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
