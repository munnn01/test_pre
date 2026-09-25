#!/usr/bin/env python
"""Upload private V2 pilot archive and push a commit-pinned V3 Kaggle shard."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ops.push_dual_codec_search import require_inactive
from ops.push_paper_runtime import require_local_commit
from ops.push_rcts_pilot import account_environment, kaggle_command, notebook

TEMPLATE = REPO / "kaggle/dual_codec_lowqp_v3_cell.sh"
KINETICS = "qktttttttttt/kineticscleaned"


def archive_slug(codec: str, sha: str) -> str:
    return f"dual-v3-pilot-cache-{codec}-{sha[:10]}"


def payload(commit: str, account: str, codec: str, shard: int,
            archive: Path) -> tuple[dict, dict, dict]:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("full commit SHA required")
    if not re.fullmatch(r"[a-z0-9]+", account):
        raise ValueError("invalid Kaggle account")
    if codec not in ("h264", "h265") or shard not in (0, 1):
        raise ValueError("unsupported codec or shard")
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    dataset_slug = archive_slug(codec, sha)
    notebook_slug = f"dual-v3-lowqp-{codec}-s{shard}"
    script = TEMPLATE.read_text(encoding="utf-8")
    for old, new in (("__REF__", commit), ("__CODEC__", codec),
                     ("__SHARD__", str(shard)), ("__ACCOUNT__", account),
                     ("__DATASET_SLUG__", dataset_slug)):
        script = script.replace(old, new)
    book = notebook(script, codec)
    book["cells"][0]["id"] = f"dual-v3-{codec}-s{shard}"
    metadata = {"id": f"{account}/{notebook_slug}", "title": notebook_slug,
                "code_file": "notebook.ipynb", "language": "python",
                "kernel_type": "notebook", "is_private": True,
                "enable_gpu": True, "enable_internet": True,
                "dataset_sources": [KINETICS, f"{account}/{dataset_slug}"],
                "kernel_sources": [], "competition_sources": [], "model_sources": []}
    dataset_metadata = {"title": dataset_slug.replace("-", " "),
        "id": f"{account}/{dataset_slug}", "licenses": [{"name": "CC0-1.0"}],
        "subtitle": "Private V2 pilot records for paired V3 development ablation",
        "description": "Derived candidate measurements on old TRAIN-calibration and VAL-dev; "
                       "not a new independent holdout. Source archive SHA-256: " + sha}
    return book, metadata, dataset_metadata


def _command(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(kaggle_command() + args, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", env=env, check=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--codec", choices=("h264", "h265"), required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--pool", type=Path, default=Path("D:/STUDY/LAB/pool.json"))
    parser.add_argument("--write-only", action="store_true")
    args = parser.parse_args()
    require_local_commit(args.commit)
    if not args.archive.is_file() or args.archive.stat().st_size < 100_000:
        raise ValueError("missing pilot archive")
    book, metadata, dataset_metadata = payload(
        args.commit, args.account, args.codec, args.shard, args.archive)
    slug = metadata["id"].split("/", 1)[1]
    target = REPO / "ops/_push" / args.account / slug
    target.mkdir(parents=True, exist_ok=True)
    (target / "notebook.ipynb").write_text(json.dumps(book), encoding="utf-8")
    (target / "kernel-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    dataset_slug = dataset_metadata["id"].split("/", 1)[1]
    data_dir = REPO / "ops/_push_datasets" / args.account / dataset_slug
    data_dir.mkdir(parents=True, exist_ok=True)
    destination = data_dir / f"dual_codec_search_v2_{args.codec}.tgz"
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != hashlib.sha256(args.archive.read_bytes()).hexdigest():
        raise ValueError("existing upload payload has different archive hash")
    if not destination.exists():
        shutil.copy2(args.archive, destination)
    (data_dir / "dataset-metadata.json").write_text(
        json.dumps(dataset_metadata, indent=2), encoding="utf-8")
    print(f"[generated] {metadata['id']} with private dataset {dataset_metadata['id']}", flush=True)
    if args.write_only:
        return
    env = account_environment(args.pool, args.account)
    env["PYTHONIOENCODING"] = "utf-8"
    require_inactive(metadata["id"], env)
    status = _command(["datasets", "status", dataset_metadata["id"]], env)
    if status.returncode == 0:
        if "ready" not in (status.stdout + status.stderr).lower():
            raise RuntimeError("existing private cache dataset is not READY")
        print(f"[dataset] existing READY {dataset_metadata['id']}", flush=True)
    else:
        output = (status.stdout + status.stderr).lower()
        if not any(word in output for word in ("404", "not found", "does not exist")):
            raise RuntimeError("cannot verify private cache dataset status")
        created = _command(["datasets", "create", "-p", str(data_dir), "-q"], env)
        message = (created.stdout + created.stderr).strip()
        print(message, flush=True)
        if created.returncode:
            raise RuntimeError("private cache dataset upload failed")
        for attempt in range(24):
            ready = _command(["datasets", "status", dataset_metadata["id"]], env)
            if ready.returncode == 0 and "ready" in (ready.stdout + ready.stderr).lower():
                break
            if attempt == 23:
                raise RuntimeError("private cache dataset did not become READY")
            time.sleep(5)
    result = _command(["kernels", "push", "-p", str(target)], env)
    output = (result.stdout + result.stderr).strip()
    print(output, flush=True)
    if result.returncode or "successfully pushed" not in output.lower():
        raise SystemExit(result.returncode or 1)


if __name__ == "__main__":
    main()
