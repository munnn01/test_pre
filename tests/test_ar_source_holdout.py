"""A fresh-source claim requires explicit IDs and no overlap."""
import json

import pytest

from ops.check_ar_source_holdout import check


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_disjoint_source_preflight(tmp_path):
    old = _write(tmp_path / "old.json", {"train": [{"source_id": "old-a"}],
                                         "val": [{"source_id": "old-b"}],
                                         "test": [{"source_id": "old-c"}]})
    fresh = _write(tmp_path / "fresh.json", {"test": [
        {"source_id": "new-a", "label": 1},
        {"source_id": "new-b", "label": 2},
    ]})
    report = check(fresh, [old])
    assert report["candidate_clips"] == 2
    assert report["candidate_sources"] == 2
    assert report["overlap_sources"] == 0


def test_overlap_or_missing_source_id_fails_closed(tmp_path):
    old = _write(tmp_path / "old.json", {"train": [{"source_id": "same"}]})
    candidate = tmp_path / "candidate.json"
    _write(candidate, {"test": [{"source_id": "same", "label": 0}]})
    with pytest.raises(ValueError, match="source overlap"):
        check(candidate, [old])
    _write(candidate, {"test": [{"path": "somewhere/same.mp4", "label": 0}]})
    with pytest.raises(ValueError, match="explicit source_id"):
        check(candidate, [old])
