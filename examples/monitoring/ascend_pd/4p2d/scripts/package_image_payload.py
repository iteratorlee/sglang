#!/usr/bin/env python3
"""Export an explicit image COPY allowlist from a committed SG tree, never a worktree."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ASSETS = "examples/monitoring/ascend_pd/4p2d"
ADAPTER = "python/sglang/srt/observability/pd_monitoring.py"
# Deployment addresses/bindings are runtime mounts, never image contents.
COPY_FILES = {"monitoring/sglang_pd_monitoring.py": ADAPTER,
              "monitoring/exporter/exporter.py": ASSETS + "/exporter/exporter.py",
              **{f"source-hooks/runtime-{role}-rank.patch": ASSETS+f"/patches/runtime-{role}-rank.patch" for role in ("prefill", "decode")}}



def read_blob(repo, commit, path):
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:{path}"])


def export(repo, commit, output):
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "--verify", commit + "^{commit}"], text=True).strip()
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("expected resolved commit SHA")
    # Read and verify everything before creating output. Dirty/untracked files
    # are never read, copied, or silently included in the image context.
    blobs = {destination:read_blob(repo,commit,source) for destination,source in COPY_FILES.items()}
    if blobs["monitoring/sglang_pd_monitoring.py"] != read_blob(repo,commit,ASSETS+"/native/sglang_pd_monitoring.py"):
        raise ValueError("tracked runtime adapter and SG module differ")
    if output.exists():
        raise ValueError("output must be a new directory")
    metadata = {"schema": 1, "adapter_commit": commit, "monitoring_assets_commit": commit,
                "adapter_sha256": hashlib.sha256(blobs["monitoring/sglang_pd_monitoring.py"]).hexdigest(),
                "author": "liyan <yan.l@pku.edu.cn>", "files": {}}
    for destination, raw in blobs.items():
        path = output / destination
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(raw)
        metadata["files"][destination] = {"source": COPY_FILES[destination], "sha256": hashlib.sha256(raw).hexdigest(), "bytes":len(raw)}
    (output / "monitoring/package-version.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sg-repo",type=Path,required=True)
    parser.add_argument("--commit",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(export(args.sg_repo,args.commit,args.output),indent=2))


if __name__ == "__main__":
    main()
