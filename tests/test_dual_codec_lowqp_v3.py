"""V3 replay rules must not inspect correctness labels at inference."""
from __future__ import annotations

import numpy as np

from ops.dual_codec_lowqp_v3 import select_lowqp


def _signals() -> dict:
    return {"kl_source": 0.0, "feature_distance": 0.0,
            "source_confidence": 0.9, "source_top1_agrees": True}


def test_low_qp_area96_ablation_keeps_original_indices() -> None:
    obs = [
        {"name": "identity128", "bpp": 1.0, "signals": {"r2plus1d_18": _signals()}},
        {"name": "area112", "bpp": 0.8, "signals": {"r2plus1d_18": _signals()}},
        {"name": "area96", "bpp": 0.7, "signals": {"r2plus1d_18": _signals()}},
    ]
    policy = {"mode": "C", "primary": {"kl": .1, "feature": .05,
              "confidence": .6}, "risk_low_qp": .1, "risk_high_qp": .2}
    risks = np.array([[0., 0.], [.03, .02], [.08, .04]])
    assert select_lowqp(obs, 30, policy, risks, "v2_frozen") == 2
    assert select_lowqp(obs, 30, policy, risks, "no_area96_low") == 1
    assert select_lowqp(obs, 30, policy, risks, "guard_area96_low") == 1
    assert select_lowqp(obs, 40, policy, risks, "no_area96_low") == 2
    assert select_lowqp(obs, 40, policy, risks, "guard_area96_low") == 2


def test_guard_requires_rate_saving_as_well_as_risk() -> None:
    obs = [
        {"name": "identity128", "bpp": 1.0, "signals": {"r2plus1d_18": _signals()}},
        {"name": "area96", "bpp": .85, "signals": {"r2plus1d_18": _signals()}},
    ]
    policy = {"mode": "C", "primary": {"kl": .1, "feature": .05,
              "confidence": .6}, "risk_low_qp": .1, "risk_high_qp": .2}
    risks = np.array([[0., 0.], [.01, .01]])
    assert select_lowqp(obs, 35, policy, risks, "v2_frozen") == 1
    assert select_lowqp(obs, 35, policy, risks, "guard_area96_low") == 0


def test_unknown_arm_fails_closed() -> None:
    import pytest
    with pytest.raises(ValueError, match="unknown arm"):
        select_lowqp([], 30, {}, np.empty((0, 2)), "new_unreviewed_arm")
