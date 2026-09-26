#!/usr/bin/env python
"""Publish a verified Kaggle V8 training archive as a new private dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ops.push_dual_codec_lowqp_v3 import _verify_owned_dataset_absent
from ops.push_rcts_pilot import account_environment, kaggle_command
from ops.v8_eval import checkpoint_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--account", default="qktttttttttt")
    parser.add_argument("--pool", type=Path, default=Path("D:/STUDY/LAB/pool.json"))
    args = parser.parse_args()
    if args.account != "qktttttttttt" or not args.archive.is_file():
        raise ValueError("expected qktttttttttt and an existing V8 archive")
    manifest, calibration, _model = checkpoint_bundle(args.archive)
    if (len(manifest["fit_ids"]) != 400 or len(manifest["cal_ids"]) != 200
            or calibration["n_cal"] != 200):
        raise ValueError("incomplete V8 TRAIN/CAL artifact")
    sha = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    slug = f"v8-restorer-h264-{sha[:12]}"
    handle = f"{args.account}/{slug}"
    directory = REPO / "ops/_push_datasets" / args.account / slug
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "v8_restorer_h264_train.tgz"
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != sha:
            raise ValueError("checkpoint upload directory contains another archive")
    else:
        shutil.copy2(args.archive, destination)
    metadata = {"id": handle, "title": slug.replace("-", " "),
                "licenses": [{"name": "CC0-1.0"}],
                "subtitle": "Private locked H264 V8 decoder-restoration checkpoint",
                "description": "TRAIN fit400/CAL200 only; selected before MC3 TEST evaluation. "
                               "Archive SHA-256: " + sha}
    (directory / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2),
                                                        encoding="utf-8")
    env = account_environment(args.pool, args.account)
    env["PYTHONIOENCODING"] = "utf-8"
    def command(parts: list[str]):
        return subprocess.run(kaggle_command() + parts, capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              env=env, check=False)
    status = command(["datasets", "status", handle])
    if status.returncode == 0:
        if "ready" not in (status.stdout + status.stderr).lower():
            raise RuntimeError("checkpoint dataset exists but is not READY")
    else:
        output = (status.stdout + status.stderr).lower()
        if not any(word in output for word in ("403", "404", "not found", "does not exist")):
            raise RuntimeError("unable to establish checkpoint dataset status")
        _verify_owned_dataset_absent(handle, env)
        created = command(["datasets", "create", "-p", str(directory), "-q"])
        if created.returncode:
            raise RuntimeError((created.stdout + created.stderr).strip())
        for attempt in range(24):
            ready = command(["datasets", "status", handle])
            if ready.returncode == 0 and "ready" in (ready.stdout + ready.stderr).lower():
                break
            if attempt == 23:
                raise RuntimeError("checkpoint dataset did not become READY")
            time.sleep(5)
    print(json.dumps({"dataset": handle, "archive_sha256": sha,
                      "selected": calibration["selected"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
