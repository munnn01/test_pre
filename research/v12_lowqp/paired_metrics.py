"""Verbatim curves/compare/summarize from hash-locked V8; no logic changes."""
import math
import numpy as np
from src.metrics.bd_rate import bd_metric,bd_rate
from .policy import QPS

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
