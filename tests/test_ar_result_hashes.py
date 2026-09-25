"""The documented aggregate hashes must match the committed V2 files."""
import hashlib

import pytest

from ops.verify_ar_result_hashes import RESULTS, verify


def test_committed_v2_result_hashes_match_readme():
    assert set(verify(RESULTS)) == {"h264_result.json", "h265_result.json"}


def test_mismatch_is_detected(tmp_path):
    entries = {}
    for codec in ("h264", "h265"):
        name = f"{codec}_result.json"
        content = codec.encode()
        (tmp_path / name).write_bytes(content)
        entries[name] = hashlib.sha256(content).hexdigest()
    (tmp_path / "README.md").write_text(
        "".join(f"- `{name}`: `{sha}`\n" for name, sha in entries.items()),
        encoding="utf-8",
    )
    assert verify(tmp_path) == entries
    (tmp_path / "h264_result.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify(tmp_path)
