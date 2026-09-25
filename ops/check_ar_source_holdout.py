#!/usr/bin/env python
"""Fail-closed source-disjointness preflight for a future AR holdout index.

Both indices must carry explicit ``source_id`` values. A filename is not proof
of source identity, so this tool does not infer source IDs from file paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _records(path: Path, splits: tuple[str, ...]) -> list[dict]:
    index = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for split in splits:
        if split not in index or not isinstance(index[split], list):
            raise ValueError(f"{path}: missing list split {split}")
        records.extend(index[split])
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("source_id"), str):
            raise ValueError(f"{path}: every record needs an explicit source_id")
        if not record["source_id"].strip():
            raise ValueError(f"{path}: source_id cannot be empty")
    return records


def check(candidate_index: Path, reference_indices: list[Path],
          candidate_split: str = "test") -> dict:
    if not reference_indices:
        raise ValueError("at least one development/history reference index is required")
    candidate = _records(candidate_index, (candidate_split,))
    if not candidate:
        raise ValueError("candidate holdout is empty")
    prior = []
    for path in reference_indices:
        index = json.loads(path.read_text(encoding="utf-8"))
        splits = tuple(name for name in ("train", "val", "test") if name in index)
        if not splits:
            raise ValueError(f"{path}: no train/val/test splits")
        prior.extend(_records(path, splits))
    prior_ids = {record["source_id"].strip() for record in prior}
    candidate_ids = {record["source_id"].strip() for record in candidate}
    overlap = candidate_ids & prior_ids
    if overlap:
        raise ValueError(f"source overlap: {len(overlap)} candidate source IDs occur in history")
    if any(not isinstance(record.get("label"), int)
           or not 0 <= record["label"] < 400 for record in candidate):
        raise ValueError("candidate labels must use integer Kinetics-400 indices 0..399")
    fingerprint = hashlib.sha256("\n".join(sorted(candidate_ids)).encode()).hexdigest()
    return {"candidate_clips": len(candidate), "candidate_sources": len(candidate_ids),
            "reference_sources": len(prior_ids), "overlap_sources": 0,
            "candidate_source_fingerprint_sha256": fingerprint,
            "scope": "source-ID disjointness against supplied references only; "
                     "historical inspection and dataset provenance still require audit"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-index", type=Path, required=True)
    parser.add_argument("--candidate-split", default="test")
    parser.add_argument("--reference-index", type=Path, action="append", required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.candidate_index, args.reference_index,
                           args.candidate_split), indent=2))


if __name__ == "__main__":
    main()
