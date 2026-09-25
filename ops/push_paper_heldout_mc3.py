#!/usr/bin/env python
"""Prepare or push a private, commit-pinned unseen-analyzer Kaggle shard."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ops.push_dual_codec_search import require_inactive
from ops.push_paper_runtime import require_local_commit
from ops.push_rcts_pilot import account_environment, kaggle_command, notebook

TEMPLATE = REPO / "kaggle/paper_heldout_mc3_cell.sh"
DATASETS = ["qktttttttttt/kineticscleaned",
            "qktttttttttt/v2-paper-cache-1000-20260924"]


def payload(commit: str, account: str, slug: str, codec: str,
            shard: int, *, cpu: bool = False) -> tuple[dict, dict]:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("full commit SHA required")
    if not re.fullmatch(r"[a-z0-9]+", account):
        raise ValueError("invalid Kaggle account")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", slug):
        raise ValueError("invalid Kaggle slug")
    if codec not in ("h264", "h265") or shard not in (0, 1):
        raise ValueError("codec must be h264/h265 and shard must be 0/1")
    script = TEMPLATE.read_text(encoding="utf-8")
    for old, new in (("__REF__", commit), ("__CODEC__", codec),
                     ("__SHARD__", str(shard))):
        script = script.replace(old, new)
    book = notebook(script, codec)
    book["cells"][0]["id"] = f"paper-mc3-{codec}-s{shard}"
    metadata = {"id": f"{account}/{slug}", "title": slug,
                "code_file": "notebook.ipynb", "language": "python",
                "kernel_type": "notebook", "is_private": True,
                "enable_gpu": not cpu, "enable_internet": True,
                "dataset_sources": DATASETS, "kernel_sources": [],
                "competition_sources": [], "model_sources": []}
    return book, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--codec", choices=("h264", "h265"), required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    parser.add_argument("--pool", type=Path, default=Path("D:/STUDY/LAB/pool.json"))
    parser.add_argument("--cpu", action="store_true",
                        help="run without a Kaggle GPU when that account has no quota")
    parser.add_argument("--write-only", action="store_true")
    args = parser.parse_args()
    require_local_commit(args.commit)
    book, metadata = payload(args.commit, args.account, args.slug,
                             args.codec, args.shard, cpu=args.cpu)
    target = REPO / "ops/_push" / args.account / args.slug
    target.mkdir(parents=True, exist_ok=True)
    (target / "notebook.ipynb").write_text(json.dumps(book), encoding="utf-8")
    (target / "kernel-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    print(f"[generated] {metadata['id']} commit={args.commit}", flush=True)
    if args.write_only:
        return
    env = account_environment(args.pool, args.account)
    env["PYTHONIOENCODING"] = "utf-8"
    require_inactive(metadata["id"], env)
    result = subprocess.run(kaggle_command() + ["kernels", "push", "-p", str(target)],
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace", env=env, check=False)
    output = (result.stdout + result.stderr).strip()
    print(output, flush=True)
    if result.returncode or "successfully pushed" not in output.lower():
        raise SystemExit(result.returncode or 1)


if __name__ == "__main__":
    main()
