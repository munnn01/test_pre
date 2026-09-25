"""No-leakage/provenance checks for the V4 development evaluator."""
from __future__ import annotations

import json

import pytest

from ops.dual_codec_search import digest
from ops.dual_codec_v4_eval import _curve, load_v4
from ops.push_v4_dual_val import payload


def test_v4_frozen_artifacts_require_matching_hashes(tmp_path) -> None:
    model = {"schema": 1, "heads": {}}
    calibration = {"selected_policy": {"mode": "identity"}}
    pilot_manifest = {"experiment": "pilot"}
    v2_frozen = {"selected_policy": {"mode": "C"}}
    frozen = {"schema": 1, "codec": "h264",
              "risk_model_sha256": digest(model),
              "calibration_sha256": digest(calibration),
              "v2_manifest_sha256": digest(pilot_manifest),
              "v2_frozen_policy_sha256": digest(v2_frozen),
              "selected_policy": calibration["selected_policy"]}
    for name, value in (("risk_model", model), ("calibration", calibration),
                        ("frozen_policy", frozen)):
        (tmp_path / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
    assert load_v4(tmp_path, "h264", pilot_manifest, v2_frozen) == (model, frozen)
    frozen["risk_model_sha256"] = "bad"
    (tmp_path / "frozen_policy.json").write_text(json.dumps(frozen), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        load_v4(tmp_path, "h264", pilot_manifest, v2_frozen)


def test_all_analyzers_use_same_selected_stream() -> None:
    rows = []
    for video in range(2):
        measurements = []
        for qp in (30, 35, 40, 45, 50):
            streams = {
                "identity128": {"bpp": 1. - qp / 100,
                                "correct": {"r2plus1d_18": True,
                                            "r3d_18": False, "mc3_18": True}},
                "blur": {"bpp": .8 - qp / 100,
                         "correct": {"r2plus1d_18": False,
                                     "r3d_18": True, "mc3_18": bool(video)}}}
            measurements.append({"qp": qp,
                                 "choices": {"v2_frozen": "identity128",
                                             "v4_frozen": "blur"},
                                 "streams": streams})
        rows.append({"sequence_id": f"v{video}", "measurements": measurements})
    assert _curve(rows, "v4_frozen", "r2plus1d_18")["30"]["top1"] == 0
    assert _curve(rows, "v4_frozen", "r3d_18")["30"]["top1"] == 1
    assert _curve(rows, "v4_frozen", "mc3_18")["30"]["top1"] == .5
    assert _curve(rows, "v4_frozen", "mc3_18")["30"]["bpp"] == .5


def test_v4_val_notebook_is_private_and_commit_pinned(tmp_path) -> None:
    archive = tmp_path / "pilot.tgz"
    archive.write_bytes(b"fixture")
    commit = "a" * 40
    book, meta = payload(commit, "vtk269", "h264", 1, archive)
    script = "".join(book["cells"][0]["source"])
    assert commit in script
    assert "__REF__" not in script
    assert "__CODEC__" not in script
    assert meta["id"] == "vtk269/dual-v4-val-h264-s1"
    assert meta["is_private"] is True
    assert meta["dataset_sources"][0] == "qktttttttttt/kineticscleaned"
