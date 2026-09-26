"""Deterministic smoke checks for the V7 temporal TRAIN pilot."""
from __future__ import annotations

import numpy as np

from ops.v7_temporal_pilot import summarize, temporal_variants


def test_constant_clip_is_unchanged() -> None:
    x = np.full((4, 8, 8, 3), 73, dtype=np.uint8)
    variants = temporal_variants(x)
    assert set(variants) == {"temporal_denoise", "temporal_unsharp"}
    assert all(np.array_equal(y, x) for y in variants.values())


def test_small_temporal_impulse_changes_without_wrapping() -> None:
    x = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    x[1] = 12
    variants = temporal_variants(x)
    assert np.all(variants["temporal_denoise"][1] == 6)
    assert np.all(variants["temporal_unsharp"][1] == 16)
    assert np.all(variants["temporal_denoise"][-1] == 0)


def test_hard_cut_is_not_temporally_mixed() -> None:
    x = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    x[2:] = 255
    assert all(np.array_equal(y, x) for y in temporal_variants(x).values())


def test_oracle_counts_only_safe_matched_rescue() -> None:
    base = {"r2plus1d_18": True, "r3d_18": True, "mc3_18": False}
    good = {"r2plus1d_18": True, "r3d_18": True, "mc3_18": True}
    bad = {"r2plus1d_18": False, "r3d_18": True, "mc3_18": True}
    row = {"measurements": [
        {"qp": qp, "v2": {"correct": base},
         "candidates": [
             {"name": "temporal_denoise", "rate_ratio_to_v2": 1.01,
              "correct": good if qp == 30 else bad},
             {"name": "temporal_unsharp", "rate_ratio_to_v2": 1.04,
              "correct": good}],
         "search": {name: [{"encode_decode_s": .1}]
                    for name in ("temporal_denoise", "temporal_unsharp")}}
        for qp in (30, 35, 40)]}
    result = summarize([row])["qps"]
    assert result["30"]["rescue_matched"] == 1
    assert result["35"]["rescue_matched"] == 0
    assert result["40"]["rescue_matched"] == 0
