#!/usr/bin/env python
"""Push commit-pinned H.264 V8 train/calibration notebook to Kaggle."""
from __future__ import annotations

import argparse
import hashlib
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
from ops.push_dual_codec_lowqp_v3 import archive_slug

TEMPLATE = REPO / "kaggle/v8_train_cell.sh"


def payload(commit: str, account: str, archive: Path) -> tuple[dict, dict]:
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or not re.fullmatch(r"[a-z0-9]+", account):
        raise ValueError("full commit SHA and lowercase Kaggle account required")
    dataset_slug = archive_slug("h264", hashlib.sha256(archive.read_bytes()).hexdigest())
    slug = "dual-v8-restorer-h264-train"
    script = TEMPLATE.read_text(encoding="utf-8")
    for old, new in (("__REF__", commit), ("__DATASET_SLUG__", dataset_slug),
                     ("__ACCOUNT__", account)):
        script = script.replace(old, new)
    book = notebook(script, "h264")
    book["cells"][0]["id"] = slug
    metadata = {"id": f"{account}/{slug}", "title": slug,
        "code_file": "notebook.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": ["qktttttttttt/kineticscleaned", f"{account}/{dataset_slug}"],
        "kernel_sources": [], "competition_sources": [], "model_sources": []}
    return book, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--pool", type=Path, default=Path("D:/STUDY/LAB/pool.json"))
    parser.add_argument("--write-only", action="store_true")
    args = parser.parse_args()
    require_local_commit(args.commit)
    if not args.archive.is_file():
        raise ValueError("pilot cache archive missing")
    book, metadata = payload(args.commit, args.account, args.archive)
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
    dataset = metadata["dataset_sources"][1]
    ready = subprocess.run(kaggle_command() + ["datasets", "status", dataset],
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env=env, check=False)
    if ready.returncode or "ready" not in (ready.stdout + ready.stderr).lower():
        raise RuntimeError(f"private pilot cache not READY: {dataset}")
    result = subprocess.run(kaggle_command() + ["kernels", "push", "-p", str(target)],
                            capture_output=True, text=True, encoding="utf-8", errors="replace",
                            env=env, check=False)
    output = (result.stdout + result.stderr).strip()
    print(output, flush=True)
    if result.returncode or "successfully pushed" not in output.lower():
        raise SystemExit(result.returncode or 1)


if __name__ == "__main__":
    main()
