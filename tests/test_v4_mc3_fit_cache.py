"""Small deterministic checks for V4 source features and Kaggle payload."""
from __future__ import annotations

import numpy as np

from ops.v4_mc3_fit_cache import source_motion_features
from ops.push_v4_mc3_fit_cache import payload


def test_static_clip_has_zero_temporal_features() -> None:
    clip = np.full((16, 128, 128, 3), 80, dtype=np.uint8)
    features = source_motion_features(clip)
    assert features["temporal_mean_abs"] == 0
    assert features["temporal_p90_abs"] == 0
    assert features["temporal_active_fraction"] == 0
    assert np.isfinite(list(features.values())).all()


def test_motion_features_are_source_only_and_finite() -> None:
    clip = np.zeros((16, 8, 8, 3), dtype=np.uint8)
    clip[8:] = 255
    features = source_motion_features(clip)
    assert features["temporal_mean_abs"] > 0
    assert features["temporal_active_fraction"] > 0
    assert all(np.isfinite(value) for value in features.values())
    assert set(features) == {"temporal_mean_abs", "temporal_p90_abs",
                             "temporal_active_fraction", "first_frame_edge_mean"}


def test_payload_is_private_and_commit_pinned(tmp_path) -> None:
    archive = tmp_path / "pilot.tgz"
    archive.write_bytes(b"pilot fixture")
    commit = "a" * 40
    book, metadata = payload(commit, "vtk269", "h264", 1, archive)
    script = "".join(book["cells"][0]["source"])
    assert commit in script
    assert "__REF__" not in script
    assert "__CODEC__" not in script
    assert metadata["id"] == "vtk269/dual-v4-mc3-fit-h264-s1"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["dataset_sources"][0] == "qktttttttttt/kineticscleaned"
