#!/usr/bin/env python
"""TRAIN-only, V2-anchored sparse repair for mc3_18.

Fit on V2 TRAIN-fit paired with V4 mc3 TRAIN-fit; calibrate on disjoint
TRAIN-calibration videos. Inference sees codec measurements and source-video
features only. It never sees correctness labels or mc3 predictions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from ops.dual_codec_lowqp_v3 import load_pilot
from ops.dual_codec_search import digest, write_json
from ops.dual_codec_v4_fit import QPS, SOURCE_FEATURES, join_records
from src.models.dual_codec_search import observations, risk_features, risk_scores, select_observations

EVENTS = ("mc3_rescue", "mc3_loss", "r2_loss", "r3_loss")
FIELDS = {"r2_loss": "correct", "r3_loss": "cross_correct"}
PREMIUM_CAP = .20
RATE_BUDGET = .02  # Each measured QP: <= 2% aggregate bpp above frozen V2.
PRIMARY_GAP = -.005  # At most one additional error per 200 calibration clips/QP.
POLICY_GRID = tuple(
    {"mode": "sparse_repair", "min_score": score, "primary_weight": weight,
     "premium_cap": PREMIUM_CAP}
    for score in (.0, .01, .02, .04, .08)
    for weight in (.25, .5, 1.0, 2.0)
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def v2_choice(obs: list[dict], qp: int, old_risk: dict, frozen: dict) -> int:
    return select_observations(obs, qp, frozen, risk_scores(obs, qp, old_risk))


def pair_features(obs: list[dict], base: int, alternative: int, qp: int,
                  source: dict) -> np.ndarray:
    """Explicit, label-free candidate-versus-V2 features."""
    a = risk_features(obs, base, qp)
    b = risk_features(obs, alternative, qp)
    extra = np.asarray([source[k] for k in SOURCE_FEATURES], dtype=np.float64)
    out = np.concatenate((a, b, b - a, extra))
    if not np.all(np.isfinite(out)):
        raise ValueError("non-finite pair features")
    return out


def alternatives(obs: list[dict], base: int, cap: float) -> list[int]:
    if base == 0:
        return []  # Never revise the identity fallback of frozen V2.
    return [j for j in range(len(obs)) if j != base
            and obs[j]["bpp"] <= obs[base]["bpp"] * (1 + cap) + 1e-12]


def _fit_head(x: np.ndarray, labels: np.ndarray) -> dict:
    positives = int(labels.sum())
    if len(np.unique(labels)) < 2:
        return {"constant": float((positives + .5) / (len(labels) + 1)),
                "positive_count": positives}
    mean = x.mean(axis=0)
    variance = ((x - mean) ** 2).mean(axis=0)
    scale = np.where(variance > 1e-24, np.sqrt(variance), 1.)
    z = np.clip((x - mean) / scale, -10, 10)
    estimator = LogisticRegression(C=.05, max_iter=3000, random_state=53)
    estimator.fit(z, labels)
    return {"mean": mean.tolist(), "scale": scale.tolist(),
            "coef": estimator.coef_[0].tolist(),
            "intercept": float(estimator.intercept_[0]),
            "positive_count": positives}


def fit(rows: list[dict], old_risk: dict, frozen: dict) -> dict:
    vectors: list[np.ndarray] = []
    labels = {event: [] for event in EVENTS}
    videos = set()
    for row in rows:
        if row["sequence_id"] in videos:
            raise ValueError("duplicate TRAIN-fit video")
        videos.add(row["sequence_id"])
        for measurement in row["measurements"]:
            qp, candidates = measurement["qp"], measurement["candidates"]
            obs = observations(candidates)
            base = v2_choice(obs, qp, old_risk, frozen)
            for j in alternatives(obs, base, PREMIUM_CAP):
                vectors.append(pair_features(obs, base, j, qp, row["source_features"]))
                b, c, identity = candidates[base], candidates[j], candidates[0]
                labels["mc3_rescue"].append(int(identity["mc3_correct"]
                                                and not b["mc3_correct"]
                                                and c["mc3_correct"]))
                labels["mc3_loss"].append(int(b["mc3_correct"] and not c["mc3_correct"]))
                for event, field in FIELDS.items():
                    labels[event].append(int(b[field] and not c[field]))
    if not vectors:
        raise ValueError("no eligible TRAIN-fit V2-to-candidate pairs")
    x = np.stack(vectors)
    return {"schema": 1, "feature_dim": int(x.shape[1]),
            "source_features": list(SOURCE_FEATURES), "events": list(EVENTS),
            "training_pairs": len(x), "fit_videos": len(videos),
            "premium_cap": PREMIUM_CAP,
            "heads": {name: _fit_head(x, np.asarray(targets, dtype=np.int8))
                      for name, targets in labels.items()},
            "note": "TRAIN-fit pairwise labels only; mc3 is a development target, not an unseen analyzer"}


def _predict(x: np.ndarray, head: dict) -> float:
    if "constant" in head:
        return float(head["constant"])
    mean = np.asarray(head["mean"])
    scale = np.asarray(head["scale"])
    if x.shape != mean.shape or np.any(scale <= 0):
        raise ValueError("invalid fitted head")
    z = np.clip((x - mean) / scale, -10, 10)
    logit = float(z @ np.asarray(head["coef"]) + head["intercept"])
    return float(1 / (1 + np.exp(-np.clip(logit, -60, 60))))


def select_repair(obs: list[dict], qp: int, source: dict, state: dict,
                  policy: dict, old_risk: dict, frozen: dict) -> int:
    """One output stream; no label, MC3 inference, or dataset identity used."""
    base = v2_choice(obs, qp, old_risk, frozen)
    if qp not in QPS or policy["mode"] == "v2_frozen":
        return base
    if (policy["mode"] != "sparse_repair" or state.get("schema") != 1
            or state.get("events") != list(EVENTS)
            or state.get("source_features") != list(SOURCE_FEATURES)):
        raise ValueError("invalid V5 state/policy")
    best = (float(policy["min_score"]), base)
    for j in alternatives(obs, base, policy["premium_cap"]):
        x = pair_features(obs, base, j, qp, source)
        p = {event: _predict(x, state["heads"][event]) for event in EVENTS}
        premium = max(0., obs[j]["bpp"] / obs[base]["bpp"] - 1)
        score = (p["mc3_rescue"] - p["mc3_loss"]
                 - policy["primary_weight"] * (p["r2_loss"] + p["r3_loss"])
                 - .1 * premium)
        if score > best[0] + 1e-12:
            best = (score, j)
    return best[1]


def _calibration_row(rows: list[dict], policy: dict, state: dict,
                     old_risk: dict, frozen: dict) -> dict:
    table = {str(q): {"bpp_v2": 0., "bpp_arm": 0.,
                       "correct_v2": {k: 0 for k in ("correct", "cross_correct", "mc3_correct")},
                       "correct_arm": {k: 0 for k in ("correct", "cross_correct", "mc3_correct")},
                       "switches": 0} for q in QPS}
    choices = Counter()
    for row in rows:
        for measurement in row["measurements"]:
            qp = measurement["qp"]
            candidates = measurement["candidates"]
            obs = observations(candidates)
            b = v2_choice(obs, qp, old_risk, frozen)
            j = select_repair(obs, qp, row["source_features"], state,
                              policy, old_risk, frozen)
            cell = table[str(qp)]
            cell["bpp_v2"] += candidates[b]["bpp"]
            cell["bpp_arm"] += candidates[j]["bpp"]
            cell["switches"] += int(b != j)
            choices[candidates[j]["name"]] += 1
            for field in cell["correct_v2"]:
                cell["correct_v2"][field] += int(candidates[b][field])
                cell["correct_arm"][field] += int(candidates[j][field])
    eligible = True
    total_mc3_gain = 0
    total_switches = 0
    for cell in table.values():
        cell["bpp_change_fraction"] = cell["bpp_arm"] / cell["bpp_v2"] - 1
        cell["top1_gap"] = {field: (cell["correct_arm"][field]
                                     - cell["correct_v2"][field]) / len(rows)
                            for field in cell["correct_v2"]}
        eligible &= (cell["bpp_change_fraction"] <= RATE_BUDGET + 1e-12
                     and cell["top1_gap"]["correct"] >= PRIMARY_GAP - 1e-12
                     and cell["top1_gap"]["cross_correct"] >= PRIMARY_GAP - 1e-12)
        total_mc3_gain += cell["correct_arm"]["mc3_correct"] - cell["correct_v2"]["mc3_correct"]
        total_switches += cell["switches"]
    return {"policy": policy, "by_qp": table, "choices": dict(choices),
            "eligible": bool(eligible), "mc3_net_correct_gain": total_mc3_gain,
            "total_switches": total_switches}


def calibrate(rows: list[dict], state: dict, old_risk: dict, frozen: dict) -> dict:
    grid = [_calibration_row(rows, {"mode": "v2_frozen"}, state, old_risk, frozen)]
    grid += [_calibration_row(rows, policy, state, old_risk, frozen)
             for policy in POLICY_GRID]
    eligible = [entry for entry in grid[1:] if entry["eligible"]
                and entry["mc3_net_correct_gain"] > 0]
    selected = max(eligible, key=lambda r: (r["mc3_net_correct_gain"],
                                             -r["total_switches"])) if eligible else grid[0]
    return {"schema": 1, "calibration_videos": len(rows),
            "selected_policy": selected["policy"], "selected_summary": selected,
            "grid": grid,
            "criterion": "TRAIN-cal: maximize MC3 correct count with <=2% bpp cost/QP and <=1 additional primary error/200 videos/QP; fallback V2 if none.",
            "warning": "Three-QP TRAIN calibration is not five-QP BD-rate or independent evidence"}


def run(archive: Path, v4_dir: Path, codec: str, out_dir: Path) -> dict:
    manifest, old_risk, old_frozen, pilot = load_pilot(
        archive, codec, stages=("fit", "calibration"))
    v4_manifest = json.loads((v4_dir / "manifest.json").read_text(encoding="utf-8"))
    if (v4_manifest.get("experiment") != "dual_v4_mc3_fit_cache"
            or v4_manifest.get("codec") != codec
            or v4_manifest.get("stage_counts") != {"fit": 400, "calibration": 200}):
        raise ValueError("wrong V4 TRAIN cache")
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    if any(shard["pilot_manifest_sha256"] != digest(manifest)
           or shard["pilot_archive_sha256"] != archive_hash
           for shard in v4_manifest["source_manifests"]):
        raise ValueError("V2/V4 source provenance differs")
    records = {stage: join_records(pilot[stage], _read_jsonl(v4_dir / f"{stage}_records.jsonl"))
               for stage in ("fit", "calibration")}
    if {r["sequence_id"] for r in records["fit"]} & {r["sequence_id"] for r in records["calibration"]}:
        raise ValueError("TRAIN-fit/calibration overlap")
    frozen_policy = old_frozen["selected_policy"]
    state = fit(records["fit"], old_risk, frozen_policy)
    calibration = calibrate(records["calibration"], state, old_risk, frozen_policy)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "risk_model.json", state)
    write_json(out_dir / "calibration.json", calibration)
    frozen = {"schema": 1, "codec": codec, "qps_modified": list(QPS),
              "risk_model_sha256": digest(state),
              "calibration_sha256": digest(calibration),
              "v2_manifest_sha256": digest(manifest),
              "v4_manifest_sha256": digest(v4_manifest),
              "v2_frozen_policy_sha256": digest(old_frozen),
              "selected_policy": calibration["selected_policy"],
              "scope": "frozen TRAIN-only development; MC3 is not an independent analyzer"}
    write_json(out_dir / "frozen_policy.json", frozen)
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--v4-dir", type=Path, required=True)
    parser.add_argument("--codec", choices=("h264", "h265"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.archive, args.v4_dir, args.codec, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
