"""Small deterministic checks for V4 source features and Kaggle payload."""
from __future__ import annotations

import numpy as np
import json
from argparse import Namespace

from ops.v4_mc3_fit_cache import source_motion_features
from ops.v4_mc3_fit_cache import merge
from ops.push_v4_mc3_fit_cache import payload
from ops.dual_codec_search import digest
from src.models.codec_search import CANDIDATES


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


def test_merge_accepts_mount_specific_index_hashes(tmp_path) -> None:
    all_ids = {"fit": [f"fit-{i}" for i in range(400)],
               "calibration": [f"cal-{i}" for i in range(200)]}
    dirs = []
    for shard in (0, 1):
        directory = tmp_path / f"shard_{shard}"
        directory.mkdir()
        sample_ids = {stage: ids[shard::2] for stage, ids in all_ids.items()}
        manifest = {"experiment": "dual_v4_mc3_fit_cache", "codec": "h264",
            "shard": shard, "qps": [30, 35, 40], "candidates": list(CANDIDATES),
            "model": "mc3_18", "stage_sample_ids": sample_ids,
            "stage_shard_fingerprints": {stage: digest(ids)
                                         for stage, ids in sample_ids.items()},
            "stage_full_fingerprints": {stage: digest(ids)
                                        for stage, ids in all_ids.items()},
            "pilot_manifest_sha256": "pilot", "pilot_archive_sha256": "archive",
            "index_sha256": f"mount-specific-{shard}", "code_commit": "commit"}
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        for stage, ids in sample_ids.items():
            (directory / f"{stage}_records.jsonl").write_text(
                "".join(json.dumps({"sequence_id": key}) + "\n" for key in ids),
                encoding="utf-8")
        dirs.append(directory)
    output = tmp_path / "merged"
    merge(Namespace(codec="h264", shard_dir=dirs, out_dir=output))
    assert len((output / "fit_records.jsonl").read_text().splitlines()) == 400
    assert len((output / "calibration_records.jsonl").read_text().splitlines()) == 200
