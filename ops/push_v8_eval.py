#!/usr/bin/env python
"""Push a commit-pinned V8 H.264 paired TEST shard after checkpoint publication."""
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

TEMPLATE = REPO / "kaggle/v8_eval_cell.sh"
KINETICS = "qktttttttttt/kineticscleaned"
V2_CACHE = "qktttttttttt/v2-paper-cache-1000-20260924"


def payload(commit: str, account: str, shard: int, checkpoint_slug: str) -> tuple[dict, dict]:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("full commit SHA required")
    if account != "qktttttttttt" or shard not in (0, 1):
        raise ValueError("private V2 TEST cache requires qktttttttttt; shard must be 0 or 1")
    if not re.fullmatch(r"v8-restorer-h264-[0-9a-f]{12}", checkpoint_slug):
        raise ValueError("checkpoint dataset slug must include 12-byte SHA prefix")
    slug = f"dual-v8-restorer-h264-test-s{shard}"
    script = TEMPLATE.read_text(encoding="utf-8")
    for old, new in (("__REF__", commit), ("__SHARD__", str(shard)),
                     ("__CHECKPOINT_SLUG__", checkpoint_slug)):
        script = script.replace(old, new)
    book = notebook(script, "h264")
    book["cells"][0]["id"] = slug
    metadata = {"id": f"{account}/{slug}", "title": slug,
                "code_file": "notebook.ipynb", "language": "python",
                "kernel_type": "notebook", "is_private": True,
                "enable_gpu": True, "enable_internet": True,
                "dataset_sources": [KINETICS, V2_CACHE,
                                    f"{account}/{checkpoint_slug}"],
                "kernel_sources": [], "competition_sources": [], "model_sources": []}
    return book, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--account", default="qktttttttttt")
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    parser.add_argument("--checkpoint-slug", required=True)
    parser.add_argument("--pool", type=Path, default=Path("D:/STUDY/LAB/pool.json"))
    parser.add_argument("--write-only", action="store_true")
    args = parser.parse_args()
    require_local_commit(args.commit)
    book, metadata = payload(args.commit, args.account, args.shard, args.checkpoint_slug)
    target = REPO / "ops/_push" / args.account / metadata["id"].split("/", 1)[1]
    target.mkdir(parents=True, exist_ok=True)
    (target / "notebook.ipynb").write_text(json.dumps(book), encoding="utf-8")
    (target / "kernel-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    print(f"[generated] {metadata['id']} commit={args.commit}", flush=True)
    if args.write_only:
        return
    env = account_environment(args.pool, args.account)
    env["PYTHONIOENCODING"] = "utf-8"
    require_inactive(metadata["id"], env)
    for source in metadata["dataset_sources"][1:]:
        ready = subprocess.run(kaggle_command() + ["datasets", "status", source],
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace", env=env, check=False)
        if ready.returncode or "ready" not in (ready.stdout + ready.stderr).lower():
            raise RuntimeError(f"private dataset not READY: {source}")
    result = subprocess.run(kaggle_command() + ["kernels", "push", "-p", str(target)],
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace", env=env, check=False)
    output = (result.stdout + result.stderr).strip()
    print(output, flush=True)
    if result.returncode or "successfully pushed" not in output.lower():
        raise SystemExit(result.returncode or 1)


if __name__ == "__main__":
    main()
