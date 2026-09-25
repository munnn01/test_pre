"""Unit checks for V4 pairing, feature isolation and selector behavior."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from ops.dual_codec_v4_fit import (ANALYZERS, OUTCOME_FIELDS, QPS,
                                   SOURCE_FEATURES, feature_vector,
                                   calibrate, fit_regret, join_records, predict_events,
                                   select_v4)
from src.models.codec_search import CANDIDATES
from src.models.dual_codec_search import MODELS, SIGNALS, observations


def _candidate(name: str, bpp: float, correct: bool) -> dict:
    signals = {model: {key: .1 for key in SIGNALS} for model in MODELS}
    return {"name": name, "bpp": bpp, "signals": signals,
            "correct": correct, "cross_correct": correct}


def _pair(video: int) -> tuple[dict, dict]:
    old_measurements, new_measurements = [], []
    for qp in QPS:
        candidates = [_candidate(name, 1 - i * .1, (video + i + qp) % 3 != 0)
                      for i, name in enumerate(CANDIDATES)]
        fresh = [{"name": c["name"], "bpp": c["bpp"],
                  "mc3_correct": (video + i + qp) % 4 != 0}
                 for i, c in enumerate(candidates)]
        old_measurements.append({"qp": qp, "candidates": candidates})
        new_measurements.append({"qp": qp, "candidates": fresh})
    source = dict.fromkeys(SOURCE_FEATURES, .05 + video / 100)
    return ({"sequence_id": f"video-{video}", "measurements": old_measurements},
            {"sequence_id": f"video-{video}", "source_features": source,
             "measurements": new_measurements})


def test_join_rejects_wrong_bitrate_and_video() -> None:
    old, fresh = _pair(1)
    assert len(join_records([old], [fresh])[0]["measurements"]) == len(QPS)
    changed = copy.deepcopy(fresh)
    changed["measurements"][0]["candidates"][1]["bpp"] += .001
    with pytest.raises(ValueError, match="bitrates differ"):
        join_records([old], [changed])
    changed = copy.deepcopy(fresh)
    changed["sequence_id"] = "another-video"
    with pytest.raises(ValueError, match="video IDs differ"):
        join_records([old], [changed])


def test_deployed_features_ignore_outcome_labels() -> None:
    old, fresh = _pair(2)
    row = join_records([old], [fresh])[0]
    measurement = row["measurements"][0]
    before = feature_vector(observations(measurement["candidates"]), 1,
                            measurement["qp"], row["source_features"])
    changed = copy.deepcopy(measurement)
    for candidate in changed["candidates"]:
        for model in ANALYZERS:
            candidate[OUTCOME_FIELDS[model]] = not candidate[OUTCOME_FIELDS[model]]
    after = feature_vector(observations(changed["candidates"]), 1,
                           measurement["qp"], row["source_features"])
    np.testing.assert_array_equal(before, after)


def test_fitted_selector_never_chooses_higher_bitrate() -> None:
    pairs = [_pair(i) for i in range(16)]
    rows = join_records([a for a, _ in pairs], [b for _, b in pairs])
    state = fit_regret(rows)
    assert state["fit_videos"] == 16
    assert set(state["heads"]) == set(ANALYZERS)
    measurement = rows[0]["measurements"][0]
    obs = observations(measurement["candidates"])
    events = predict_events(obs, measurement["qp"], rows[0]["source_features"], state)
    assert all(0 <= events[m]["harm"][1] <= 1 for m in ANALYZERS)
    # The legacy state is not consulted for a V4-controlled critical QP.
    choice = select_v4(obs, measurement["qp"], rows[0]["source_features"],
                       state, {"mode": "v4", "low_qp_harm": 1.,
                               "qp40_harm": 1., "gain_weight": 0.}, {}, {})
    assert 0 <= choice < len(CANDIDATES)
    assert obs[choice]["bpp"] <= obs[0]["bpp"]


def test_calibration_reports_three_qp_gaps_without_test_data() -> None:
    pairs = [_pair(i) for i in range(16)]
    rows = join_records([a for a, _ in pairs], [b for _, b in pairs])
    state = fit_regret(rows)
    legacy_risk = {"schema": 1, "models": {model: {"constant": 0.}
                                            for model in MODELS}}
    report = calibrate(rows, state, legacy_risk, {"mode": "identity"})
    assert report["calibration_videos"] == 16
    assert len(report["grid"]) == 2 + 8 * 3
    assert set(report["selected_summary"]["same_qp_top1_gaps"]) == set(ANALYZERS)
    assert set(report["selected_summary"]["same_qp_top1_gaps"]["mc3_18"]) == {
        str(qp) for qp in QPS}
