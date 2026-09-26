"""Check rate matching and the label-aware TRAIN feasibility accounting."""
from __future__ import annotations

import numpy as np

from ops.v6_rate_matched_pilot import choose_qps, encode_grid, qp_grid, summarize


def test_qp_grid_respects_codec_bounds():
    assert qp_grid(30) == (26, 30, 34, 38, 42, 46)
    assert qp_grid(40) == (36, 40, 44, 48)


def test_choose_qps_keeps_nearest_under_budget_and_nearest_over():
    probes = {32: {"bpp": 1.08}, 33: {"bpp": 1.03},
              34: {"bpp": 1.01}, 35: {"bpp": .96}}
    assert choose_qps(probes, 1.) == [34]
    probes[34]["bpp"] = 1.04
    assert choose_qps(probes, 1.) == [33, 35]


def test_codec_search_checks_integer_qps_near_byte_target():
    class FakeCodec:
        def _encode_decode_clip(self, video, qp):
            _, h, w, _ = video.shape
            reference_bpp = 1.5 - .05 * (qp - 35)
            return video.copy(), reference_bpp * 128 * 128 / (h * w)

    video = np.zeros((2, 96, 96, 3), dtype=np.uint8)
    shortlisted, probes = encode_grid(video, FakeCodec(), 35, 1.)
    assert 45 in shortlisted
    assert any(p["qp"] == 45 and abs(p["bpp"] - 1.) < 1e-9 for p in probes)


def test_oracle_counts_only_rate_matched_mc3_rescue_with_primary_safety():
    base = {"bpp": 1., "correct": {
        "r2plus1d_18": True, "r3d_18": True, "mc3_18": False}}
    safe = {"resolution": 128, "bpp": 1.01, "rate_ratio_to_v2": 1.01,
        "correct": {"r2plus1d_18": True, "r3d_18": True, "mc3_18": True}}
    harmful = {"resolution": 112, "bpp": .99, "rate_ratio_to_v2": .99,
        "correct": {"r2plus1d_18": False, "r3d_18": True, "mc3_18": True}}
    far = {"resolution": 96, "bpp": .8, "rate_ratio_to_v2": .8,
        "correct": {"r2plus1d_18": True, "r3d_18": True, "mc3_18": True}}
    row = {"measurements": [{"qp": 30, "v2": base,
        "candidates": [safe, harmful, far],
        "search": {"96": [], "112": [], "128": []}}]}
    result = summarize([row])["qps"]["30"]
    assert result["v2_mc3_wrong"] == 1
    assert result["matched_any"] == 1
    assert result["rescue_matched"] == 1
    assert result["rescue_by_resolution"] == {"96": 0, "112": 0, "128": 1}
    assert abs(result["oracle_mean_bpp_change_pct"] - 1.) < 1e-9
