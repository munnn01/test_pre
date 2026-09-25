#!/usr/bin/env python
"""Fit and calibrate a codec-specific three-analyzer V4 selector.

The only supervision is the V2 TRAIN-fit cache plus the V4 mc3 TRAIN-fit
cache. Thresholds are selected on disjoint TRAIN-calibration clips. This
command never reads V2 DEV, VAL, or TEST outcomes, and does not report a
generalization result.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ops.dual_codec_lowqp_v3 import load_pilot
from ops.dual_codec_search import digest, write_json
from src.models.codec_search import CANDIDATES
from src.models.dual_codec_search import MODELS, observations, risk_features, risk_scores, select_observations

QPS = (30, 35, 40)
ANALYZERS = (*MODELS, "mc3_18")
SOURCE_FEATURES = ("temporal_mean_abs", "temporal_p90_abs",
                   "temporal_active_fraction", "first_frame_edge_mean")
OUTCOME_FIELDS = {MODELS[0]: "correct", MODELS[1]: "cross_correct",
                  "mc3_18": "mc3_correct"}
# Deliberately small, declared before any V4 calibration output is inspected.
GATES = ((.04, .04), (.08, .04), (.08, .08), (.12, .08),
         (.12, .12), (.20, .12), (.20, .20), (.30, .20))
GAIN_WEIGHTS = (0., .5, 1.)
MIN_QP_ACCURACY_GAP = -.01


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def join_records(pilot: list[dict], mc3: list[dict]) -> list[dict]:
    """Pair each original video/QP/candidate with its V4 TRAIN label."""
    if len(pilot) != len(mc3):
        raise ValueError("V2/V4 source counts differ")
    result = []
    for old, new in zip(pilot, mc3):
        if old["sequence_id"] != new["sequence_id"]:
            raise ValueError("V2/V4 video IDs differ")
        source = new["source_features"]
        if set(source) != set(SOURCE_FEATURES) or not all(
                np.isfinite(float(source[k])) for k in SOURCE_FEATURES):
            raise ValueError("invalid source-only features")
        measurements = []
        old_by_qp = {item["qp"]: item for item in old["measurements"]}
        if [item["qp"] for item in new["measurements"]] != list(QPS):
            raise ValueError("V4 critical QPs are incomplete")
        for fresh in new["measurements"]:
            cached = old_by_qp[fresh["qp"]]
            if ([c["name"] for c in cached["candidates"]] != list(CANDIDATES)
                    or [c["name"] for c in fresh["candidates"]] != list(CANDIDATES)):
                raise ValueError("candidate order differs")
            candidates = []
            for prev, current in zip(cached["candidates"], fresh["candidates"]):
                if abs(prev["bpp"] - current["bpp"]) > 1e-9:
                    raise ValueError("V2/V4 bitrates differ")
                if not isinstance(current["mc3_correct"], bool):
                    raise ValueError("missing mc3 TRAIN label")
                candidates.append({**prev, "mc3_correct": current["mc3_correct"]})
            measurements.append({"qp": fresh["qp"], "candidates": candidates})
        result.append({"sequence_id": old["sequence_id"],
                       "source_features": source, "measurements": measurements})
    return result


def feature_vector(obs: list[dict], index: int, qp: int,
                   source: dict) -> np.ndarray:
    """Strict inference allowlist: two old analyzer signals plus raw-video stats."""
    return np.concatenate((risk_features(obs, index, qp),
                           np.asarray([source[k] for k in SOURCE_FEATURES], dtype=np.float64)))


def _fit_binary(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> dict:
    count = int(y.sum())
    if len(np.unique(y)) < 2:
        # A constant empirical probability is more honest than an invented fit.
        return {"constant": float((count + .5) / (len(y) + 1)),
                "positive_count": count}
    scaler = StandardScaler().fit(x, sample_weight=weight)
    z = np.clip(scaler.transform(x), -10, 10)
    estimator = LogisticRegression(C=.1, max_iter=2000, random_state=53)
    estimator.fit(z, y, sample_weight=weight * (len(y) / weight.sum()))
    return {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "coef": estimator.coef_[0].tolist(),
            "intercept": float(estimator.intercept_[0]),
            "positive_count": count}


def fit_regret(rows: list[dict]) -> dict:
    x, weights = [], []
    labels = {model: {event: [] for event in ("harm", "gain")}
              for model in ANALYZERS}
    for row in rows:
        for measurement in row["measurements"]:
            obs = observations(measurement["candidates"])
            valid = [i for i in range(1, len(obs)) if obs[i]["bpp"] < obs[0]["bpp"]]
            if not valid:
                continue
            for i in valid:
                x.append(feature_vector(obs, i, measurement["qp"], row["source_features"]))
                weights.append(1 / (len(valid) * len(QPS)))
                for model in ANALYZERS:
                    field = OUTCOME_FIELDS[model]
                    anchor = measurement["candidates"][0][field]
                    chosen = measurement["candidates"][i][field]
                    labels[model]["harm"].append(int(anchor and not chosen))
                    labels[model]["gain"].append(int(not anchor and chosen))
    if not x:
        raise ValueError("no bitrate-saving candidates in TRAIN-fit")
    matrix = np.stack(x)
    weight = np.asarray(weights)
    heads = {model: {event: _fit_binary(matrix, np.asarray(values), weight)
                     for event, values in outcomes.items()}
             for model, outcomes in labels.items()}
    return {"schema": 1, "analyzers": list(ANALYZERS),
            "source_features": list(SOURCE_FEATURES), "fit_videos": len(rows),
            "training_rows": len(matrix), "feature_dim": matrix.shape[1],
            "heads": heads, "regularization_C": .1,
            "note": "TRAIN-fit only; three analyzer harm/gain labels, no deployed mc3 signal"}


def _probability(x: np.ndarray, head: dict) -> np.ndarray:
    if "constant" in head:
        return np.full(len(x), head["constant"])
    mean, scale = np.asarray(head["mean"]), np.asarray(head["scale"])
    if x.shape[1] != len(mean) or np.any(scale <= 0):
        raise ValueError("invalid fitted feature schema")
    z = np.clip((x - mean) / scale, -10, 10) @ np.asarray(head["coef"])
    z += head["intercept"]
    return 1 / (1 + np.exp(-np.clip(z, -60, 60)))


def predict_events(obs: list[dict], qp: int, source: dict,
                   state: dict) -> dict[str, dict[str, np.ndarray]]:
    if (state.get("schema") != 1 or state.get("analyzers") != list(ANALYZERS)
            or state.get("source_features") != list(SOURCE_FEATURES)):
        raise ValueError("unexpected V4 regret model")
    x = np.stack([feature_vector(obs, i, qp, source) for i in range(len(obs))])
    result = {model: {event: _probability(x, state["heads"][model][event])
                      for event in ("harm", "gain")} for model in ANALYZERS}
    for model in ANALYZERS:
        for event in ("harm", "gain"):
            result[model][event][0] = 0
    return result


def select_v4(obs: list[dict], qp: int, source: dict, state: dict,
              policy: dict, old_risk: dict, frozen: dict) -> int:
    """For QP45/50, preserve V2 exactly; otherwise use label-free V4 heads."""
    if qp not in QPS:
        return select_observations(obs, qp, frozen, risk_scores(obs, qp, old_risk))
    if policy["mode"] == "identity":
        return 0
    if policy["mode"] != "v4":
        raise ValueError("invalid V4 policy")
    events = predict_events(obs, qp, source, state)
    threshold = policy["low_qp_harm"] if qp <= 35 else policy["qp40_harm"]
    allowed = [0]
    for i in range(1, len(obs)):
        if obs[i]["bpp"] >= obs[0]["bpp"]:
            continue
        if all(events[m]["harm"][i] <= threshold for m in ANALYZERS):
            allowed.append(i)
    def utility(i: int) -> float:
        saving = 1 - obs[i]["bpp"] / obs[0]["bpp"]
        net = np.mean([events[m]["gain"][i] - events[m]["harm"][i]
                       for m in ANALYZERS])
        return float(saving + policy["gain_weight"] * net)
    return max(allowed, key=lambda i: (utility(i), -i))


def calibration_summary(rows: list[dict], state: dict, policy: dict,
                        old_risk: dict, frozen: dict) -> dict:
    totals = {model: {str(qp): {"anchor": 0, "selected": 0}
                      for qp in QPS} for model in ANALYZERS}
    bpp_anchor, bpp_selected = [], []
    choices = Counter()
    for row in rows:
        for measurement in row["measurements"]:
            qp = measurement["qp"]
            obs = observations(measurement["candidates"])
            if policy["mode"] == "v2_frozen":
                i = select_observations(obs, qp, frozen,
                                        risk_scores(obs, qp, old_risk))
            else:
                i = select_v4(obs, qp, row["source_features"], state,
                              policy, old_risk, frozen)
            anchor, selected = measurement["candidates"][0], measurement["candidates"][i]
            bpp_anchor.append(anchor["bpp"])
            bpp_selected.append(selected["bpp"])
            choices[selected["name"]] += 1
            for model in ANALYZERS:
                field = OUTCOME_FIELDS[model]
                totals[model][str(qp)]["anchor"] += int(anchor[field])
                totals[model][str(qp)]["selected"] += int(selected[field])
    gaps = {model: {key: (value["selected"] - value["anchor"]) / len(rows)
                    for key, value in by_qp.items()} for model, by_qp in totals.items()}
    saving = 1 - float(np.mean(bpp_selected) / np.mean(bpp_anchor))
    eligible = all(gap >= MIN_QP_ACCURACY_GAP - 1e-12
                   for by_qp in gaps.values() for gap in by_qp.values())
    return {"policy": policy, "n_videos": len(rows),
            "mean_bpp_saving_fraction": saving,
            "same_qp_top1_gaps": gaps, "counts": totals,
            "choices": dict(choices), "eligible": eligible}


def calibrate(rows: list[dict], state: dict, old_risk: dict,
              frozen: dict) -> dict:
    policies = [{"mode": "identity"}, {"mode": "v2_frozen"}]
    policies.extend({"mode": "v4", "low_qp_harm": low,
                     "qp40_harm": mid, "gain_weight": weight}
                    for low, mid in GATES for weight in GAIN_WEIGHTS)
    grid = [calibration_summary(rows, state, policy, old_risk, frozen)
            for policy in policies]
    candidates = [row for row in grid if row["policy"]["mode"] == "v4"
                  and row["eligible"]]
    chosen = max(candidates,
                 key=lambda row: (row["mean_bpp_saving_fraction"],
                                  min(min(v.values()) for v in row["same_qp_top1_gaps"].values()))) if candidates else grid[0]
    return {"schema": 1, "calibration_videos": len(rows),
            "criterion": "Each of 3 analyzers at QP30/35/40 >= identity minus 1 percentage point; then maximize mean bpp saving.",
            "selected_policy": chosen["policy"], "selected_summary": chosen,
            "grid": grid, "warning": "TRAIN calibration, not independent evidence or full 5-QP BD-rate"}


def run(archive: Path, v4_dir: Path, codec: str, out_dir: Path) -> dict:
    pilot_manifest, old_risk, old_frozen, pilot = load_pilot(
        archive, codec, stages=("fit", "calibration"))
    v4_manifest = json.loads((v4_dir / "manifest.json").read_text(encoding="utf-8"))
    if (v4_manifest.get("experiment") != "dual_v4_mc3_fit_cache"
            or v4_manifest.get("codec") != codec
            or v4_manifest.get("stage_counts") != {"fit": 400, "calibration": 200}):
        raise ValueError("wrong merged V4 TRAIN cache")
    for shard in v4_manifest["source_manifests"]:
        if (shard["pilot_manifest_sha256"] != digest(pilot_manifest)
                or shard["pilot_archive_sha256"] != hashlib.sha256(archive.read_bytes()).hexdigest()):
            raise ValueError("V2/V4 provenance differs")
    records = {stage: join_records(pilot[stage], _read_jsonl(v4_dir / f"{stage}_records.jsonl"))
               for stage in ("fit", "calibration")}
    if set(r["sequence_id"] for r in records["fit"]) & set(
            r["sequence_id"] for r in records["calibration"]):
        raise ValueError("TRAIN fit/calibration overlap")
    model = fit_regret(records["fit"])
    calibration = calibrate(records["calibration"], model, old_risk,
                            old_frozen["selected_policy"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "risk_model.json", model)
    write_json(out_dir / "calibration.json", calibration)
    frozen = {"schema": 1, "codec": codec, "qps_modified": list(QPS),
              "risk_model_sha256": digest(model),
              "calibration_sha256": digest(calibration),
              "v2_manifest_sha256": digest(pilot_manifest),
              "v4_manifest_sha256": digest(v4_manifest),
              "v2_frozen_policy_sha256": digest(old_frozen),
              "selected_policy": calibration["selected_policy"],
              "scope": "development policy frozen using TRAIN fit/calibration only"}
    write_json(out_dir / "frozen_policy.json", frozen)
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--v4-dir", type=Path, required=True)
    parser.add_argument("--codec", choices=("h264", "h265"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.archive, args.v4_dir, args.codec, args.out_dir)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
