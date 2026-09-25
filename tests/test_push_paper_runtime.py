"""Runtime Kaggle notebook creation must be private and commit-pinned."""
import pytest

from ops.push_paper_runtime import payload


SHA = "a" * 40


def test_payload_pins_source_and_dataset():
    book, metadata = payload(SHA, "qktttttttttt", "paper-runtime-h264-qp40",
                             "h264", "40", 20)
    cell = book["cells"][0]["source"]
    source = "".join(cell) if isinstance(cell, list) else cell
    assert SHA in source
    assert "github.com/munnn01/test_pre.git" in source
    assert 'CODEC="h264"' in source
    assert 'QPS="40"' in source
    assert metadata["is_private"] is True
    assert metadata["dataset_sources"] == ["qktttttttttt/kineticscleaned"]


def test_payload_rejects_unpinned_or_invalid_settings():
    with pytest.raises(ValueError, match="full commit SHA"):
        payload("main", "qktttttttttt", "runtime", "h264", "40", 20)
    with pytest.raises(ValueError, match="QPs"):
        payload(SHA, "qktttttttttt", "runtime", "h264", "40,40", 20)
