"""Small inference invariants for V2-anchored MC3 repair."""
from __future__ import annotations

import numpy as np

from ops import dual_codec_v5_repair as repair


def test_alternatives_are_relative_to_v2_and_skip_identity_fallback():
    obs = [{"bpp": 1.0}, {"bpp": .5}, {"bpp": .59}, {"bpp": .7}]
    assert repair.alternatives(obs, 0, .2) == []
    assert repair.alternatives(obs, 1, .2) == [2]


def test_sparse_selector_changes_only_allowed_low_qp(monkeypatch):
    obs = [{"bpp": 1.0}, {"bpp": .5}, {"bpp": .55}, {"bpp": .8}]
    monkeypatch.setattr(repair, "v2_choice", lambda *_: 1)
    monkeypatch.setattr(repair, "pair_features", lambda *_: np.zeros(1))
    state = {"schema": 1, "events": list(repair.EVENTS),
             "source_features": list(repair.SOURCE_FEATURES),
             "heads": {"mc3_rescue": {"constant": .8},
                       "mc3_loss": {"constant": .01},
                       "r2_loss": {"constant": .01},
                       "r3_loss": {"constant": .01}}}
    policy = {"mode": "sparse_repair", "min_score": .1,
              "primary_weight": 1., "premium_cap": .2}
    assert repair.select_repair(obs, 30, {}, state, policy, {}, {}) == 2
    assert repair.select_repair(obs, 45, {}, state, policy, {}, {}) == 1
    assert repair.select_repair(obs, 30, {}, state, {"mode": "v2_frozen"}, {}, {}) == 1


def test_observations_drop_correctness_labels():
    from src.models.dual_codec_search import observations, SIGNALS, MODELS
    from src.models.codec_search import CANDIDATES
    candidates = []
    for name in CANDIDATES:
        candidates.append({"name": name, "bpp": 1., "correct": True,
            "cross_correct": True, "mc3_correct": False,
            "signals": {model: {key: 0. for key in SIGNALS} for model in MODELS}})
    obs = observations(candidates)
    assert all("correct" not in row and "mc3_correct" not in row for row in obs)
