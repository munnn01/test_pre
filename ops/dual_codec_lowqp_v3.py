#!/usr/bin/env python
"""Replay preregistered low-QP V3 ablations on the V2 pilot cache.

This is a development experiment, not a fresh source-disjoint holdout.  The
mc3_18 result on the old TEST motivated these arms, but mc3 outputs and any
ground-truth label remain unavailable to the selector.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

from ops.codec_search_ar import QPS
from ops.dual_codec_search import bootstrap, digest, metrics, prepare, validate_row, write_json
from src.models.codec_search import CANDIDATES
from src.models.dual_codec_search import MODELS, select_observations

REPO = Path(__file__).resolve().parents[1]
ARMS = ("v2_frozen", "no_area96_low", "guard_area96_low")
LOW_QPS = (30, 35)
GUARD_RISK_MULTIPLIER = 0.5
GUARD_MIN_RATE_SAVING = 0.20


def _read_json(bundle: tarfile.TarFile, name: str) -> dict:
    member = bundle.getmember(name)
    if not member.isfile() or member.size > 20_000_000:
        raise ValueError(f"unexpected cache member: {name}")
    stream = bundle.extractfile(member)
    if stream is None:
        raise ValueError(f"missing cache member: {name}")
    return json.load(stream)


def load_pilot(archive: Path, codec: str,
               stages: tuple[str, ...] = ("calibration", "dev")) -> tuple[dict, dict, dict, dict[str, list[dict]]]:
    root = f"outputs/dual_codec_search_v2/{codec}"
    with tarfile.open(archive, "r:gz") as bundle:
        manifest = _read_json(bundle, f"{root}/manifest.json")
        risk = _read_json(bundle, f"{root}/risk_model.json")
        frozen = _read_json(bundle, f"{root}/frozen_policy.json")
        if (manifest.get("experiment") != "dual_codec_search_v2_pilot"
                or manifest.get("codec") != codec
                or manifest.get("qps") != list(QPS)
                or manifest.get("candidates") != list(CANDIDATES)
                or manifest.get("models") != list(MODELS)
                or frozen.get("risk_sha256") != digest(risk)
                or frozen.get("manifest_sha256") != digest(manifest)
                or frozen.get("selected_policy") != frozen.get("policies", {}).get("C")
                or frozen["selected_policy"].get("mode") != "C"):
            raise ValueError("archive does not contain the expected frozen V2 pilot")
        # The report is pinned to the exact V2 policy and risk state in this repo.
        local = REPO / "configs/dual_codec_search_v2_frozen" / codec
        if (json.loads((local / "frozen_policy.json").read_text(encoding="utf-8")) != frozen
                or json.loads((local / "risk_model.json").read_text(encoding="utf-8")) != risk):
            raise ValueError("pilot archive disagrees with repository frozen artifacts")
        records = {}
        expected_counts = {"fit": 400, "calibration": 200, "dev": 200}
        if not stages or any(stage not in expected_counts for stage in stages):
            raise ValueError("unsupported pilot stage request")
        for stage in stages:
            expected = expected_counts[stage]
            prefix = f"{root}/cache/{stage}/clip_"
            names = sorted(name for name in bundle.getnames()
                           if name.startswith(prefix) and name.endswith(".json"))
            if len(names) != expected or len(manifest["split_ids"][stage]) != expected:
                raise ValueError(f"incomplete {stage} pilot cache")
            by_id = {}
            for name in names:
                row = _read_json(bundle, name)
                key = row["sequence_id"]
                if key in by_id:
                    raise ValueError(f"duplicate source in {stage}: {key}")
                validate_row(row, key, codec, digest(manifest))
                by_id[key] = row
            if set(by_id) != set(manifest["split_ids"][stage]):
                raise ValueError(f"{stage} IDs disagree with pilot manifest")
            records[stage] = [by_id[key] for key in manifest["split_ids"][stage]]
    return manifest, risk, frozen, records


def select_lowqp(obs: list[dict], qp: int, policy: dict,
                 risks: np.ndarray, arm: str) -> int:
    """Select using only bitrate, QP, and V2 analyzer signals/risk scores."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if qp not in LOW_QPS or arm == "v2_frozen":
        return select_observations(obs, qp, policy, risks)
    allowed = [0]
    threshold = policy["risk_low_qp"]
    for index, candidate in enumerate(obs[1:], 1):
        if candidate["name"] == "area96":
            if arm == "no_area96_low":
                continue
            saving = 1 - candidate["bpp"] / obs[0]["bpp"]
            if (saving < GUARD_MIN_RATE_SAVING
                    or not np.all(np.isfinite(risks[index]))
                    or np.any(risks[index] > threshold * GUARD_RISK_MULTIPLIER)):
                continue
        allowed.append(index)
    choice = select_observations([obs[i] for i in allowed], qp, policy,
                                 risks[allowed])
    return allowed[choice]


def selected_arrays_v3(prepared: list, policy: dict, arm: str) -> tuple[np.ndarray, dict]:
    values, choices = [], Counter()
    for clip in prepared:
        points = []
        for measurement, obs, risks in clip:
            index = select_lowqp(obs, measurement["qp"], policy, risks, arm)
            candidate = measurement["candidates"][index]
            points.append([candidate["bpp"], float(candidate["correct"]),
                           float(candidate["cross_correct"])])
            choices[candidate["name"]] += 1
        values.append(points)
    return np.asarray(values), dict(choices)


def run(archive: Path, codec: str, out: Path, draws: int) -> dict:
    manifest, risk, frozen, records = load_pilot(archive, codec)
    policy = frozen["selected_policy"]
    report = {
        "experiment": "dual_codec_lowqp_v3_development_replay",
        "interpretation": "Exploratory V3 ablation on the old V2 pilot CAL/DEV. Not a new holdout.",
        "codec": codec,
        "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "source_manifest_sha256": digest(manifest),
        "frozen_policy_sha256": digest(frozen),
        "split_fingerprints": manifest["split_fingerprints"],
        "bootstrap_unit": "whole video, all five QPs paired",
        "preregistered_arms": list(ARMS),
        "low_qps": list(LOW_QPS),
        "guard_risk_multiplier": GUARD_RISK_MULTIPLIER,
        "guard_min_rate_saving": GUARD_MIN_RATE_SAVING,
        "stages": {},
    }
    for stage in ("calibration", "dev"):
        prepared = prepare(records[stage], risk)
        identity = {"mode": "identity"}
        base, _ = selected_arrays_v3(prepared, identity, "v2_frozen")
        arms = {}
        for arm in ARMS:
            trial, choices = selected_arrays_v3(prepared, policy, arm)
            arms[arm] = {"choices": choices, "analyzers": metrics(base, trial),
                         "bootstrap": bootstrap(base, trial, draws)}
        report["stages"][stage] = {"n_videos": len(records[stage]), "arms": arms}
    write_json(out, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--codec", choices=("h264", "h265"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    if args.bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    report = run(args.archive, args.codec, args.out, args.bootstrap)
    compact = {stage: {arm: {model: data["metrics"]
                             for model, data in result["analyzers"].items()}
                       for arm, result in entry["arms"].items()}
               for stage, entry in report["stages"].items()}
    print(json.dumps({"codec": args.codec, "results": compact}, indent=2), flush=True)


if __name__ == "__main__":
    main()
