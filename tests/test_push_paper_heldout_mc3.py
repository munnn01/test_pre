"""The mc3 transfer notebook must be private, frozen and cache-backed."""
import pytest

from ops.push_paper_heldout_mc3 import DATASETS, payload


SHA = "a" * 40


def test_payload_uses_frozen_commit_and_private_cache():
    book, metadata = payload(SHA, "qktttttttttt", "paper-mc3-h264-s0", "h264", 0)
    cell = book["cells"][0]["source"]
    source = "".join(cell) if isinstance(cell, list) else cell
    assert f'REF="{SHA}"' in source
    assert 'CODEC="h264"' in source
    assert 'SHARD="0"' in source
    assert "github.com/munnn01/test_pre.git" in source
    assert "ops.paper_heldout_mc3 evaluate" in source
    assert metadata["dataset_sources"] == DATASETS
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True


def test_payload_rejects_unpinned_and_wrong_shard():
    with pytest.raises(ValueError, match="full commit"):
        payload("main", "qktttttttttt", "paper-mc3", "h264", 0)
    with pytest.raises(ValueError, match="shard"):
        payload(SHA, "qktttttttttt", "paper-mc3", "h265", 2)


def test_cpu_mode_removes_gpu_requirement():
    book, metadata = payload(SHA, "qktttttttttt", "paper-mc3-h265-s1",
                             "h265", 1, cpu=True)
    source = "".join(book["cells"][0]["source"])
    assert metadata["enable_gpu"] is False
    assert 'SHARD="1"' in source
    assert "GPU is required" not in source
