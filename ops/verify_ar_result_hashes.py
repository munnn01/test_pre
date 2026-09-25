#!/usr/bin/env python
"""Check the committed V2 aggregate files against their documented SHA-256s."""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results/dual_codec_search_v2_confirm_1000"
CODECS = ("h264", "h265")


def verify(results_dir: Path) -> dict[str, str]:
    readme = (results_dir / "README.md").read_text(encoding="utf-8")
    verified = {}
    for codec in CODECS:
        name = f"{codec}_result.json"
        matches = re.findall(
            rf"(?m)^- `{re.escape(name)}`: `([0-9a-f]{{64}})`\s*$", readme
        )
        if len(matches) != 1:
            raise ValueError(f"expected exactly one documented SHA-256 for {name}")
        actual = hashlib.sha256((results_dir / name).read_bytes()).hexdigest()
        if actual != matches[0]:
            raise ValueError(f"{name}: SHA-256 mismatch: documented={matches[0]} actual={actual}")
        verified[name] = actual
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    for name, sha in verify(args.results_dir).items():
        print(f"[verified] {name} sha256={sha}")


if __name__ == "__main__":
    main()
