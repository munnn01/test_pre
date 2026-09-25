#!/usr/bin/env python
"""Push one private, commit-pinned Kaggle runtime benchmark notebook."""
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
from ops.push_rcts_pilot import account_environment, kaggle_command, notebook

TEMPLATE = REPO / "kaggle/paper_runtime_cell.sh"
DATASET = "qktttttttttt/kineticscleaned"


def require_local_commit(commit: str) -> None:
    """Reject syntactically valid but nonexistent commit IDs before upload."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("full commit SHA required")
    result = subprocess.run(["git", "rev-parse", "--verify", f"{commit}^{{commit}}"],
                            cwd=REPO, capture_output=True, text=True, check=False)
    if result.returncode or result.stdout.strip() != commit:
        raise ValueError(f"commit is not present in this checkout: {commit}")


def payload(commit: str, account: str, slug: str, codec: str, qps: str,
            clips: int) -> tuple[dict, dict]:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("full commit SHA required")
    if not re.fullmatch(r"[a-z0-9]+", account):
        raise ValueError("invalid Kaggle account")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", slug):
        raise ValueError("invalid Kaggle slug")
    if codec not in ("h264", "h265"):
        raise ValueError("unsupported codec")
    allowed = (30, 35, 40, 45, 50)
    try:
        points = tuple(int(value) for value in qps.split(","))
    except ValueError as exc:
        raise ValueError("invalid QP list") from exc
    if not points or len(set(points)) != len(points) or any(qp not in allowed for qp in points):
        raise ValueError("QPs must be a nonempty unique subset of 30,35,40,45,50")
    if clips < 1:
        raise ValueError("clips must be positive")
    script = TEMPLATE.read_text(encoding="utf-8")
    for old, new in (("__REF__", commit), ("__CODEC__", codec),
                     ("__QPS__", ",".join(map(str, points))),
                     ("__CLIPS__", str(clips))):
        script = script.replace(old, new)
    book = notebook(script, codec)
    book["cells"][0]["id"] = f"paper-runtime-{codec}"
    metadata = {"id": f"{account}/{slug}", "title": slug,
                "code_file": "notebook.ipynb", "language": "python",
                "kernel_type": "notebook", "is_private": True,
                "enable_gpu": True, "enable_internet": True,
                "dataset_sources": [DATASET], "kernel_sources": [],
                "competition_sources": [], "model_sources": []}
    return book, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--codec", choices=("h264", "h265"), required=True)
    parser.add_argument("--qps", default="40")
    parser.add_argument("--clips", type=int, default=20)
    parser.add_argument("--pool", type=Path, default=Path("D:/STUDY/LAB/pool.json"))
    parser.add_argument("--write-only", action="store_true")
    args = parser.parse_args()
    require_local_commit(args.commit)
    book, meta = payload(args.commit, args.account, args.slug,
                         args.codec, args.qps, args.clips)
    target = REPO / "ops/_push" / args.account / args.slug
    target.mkdir(parents=True, exist_ok=True)
    (target / "notebook.ipynb").write_text(json.dumps(book), encoding="utf-8")
    (target / "kernel-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    print(f"[generated] {meta['id']} commit={args.commit}", flush=True)
    if args.write_only:
        return
    env = account_environment(args.pool, args.account)
    env["PYTHONIOENCODING"] = "utf-8"
    require_inactive(meta["id"], env)
    result = subprocess.run(kaggle_command() + ["kernels", "push", "-p", str(target)],
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace", env=env, check=False)
    output = (result.stdout + result.stderr).strip()
    print(output, flush=True)
    if result.returncode or "successfully pushed" not in output.lower():
        raise SystemExit(result.returncode or 1)


if __name__ == "__main__":
    main()
