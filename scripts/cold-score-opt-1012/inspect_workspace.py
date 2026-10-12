#!/usr/bin/env python3
"""Read private compilation metadata and installed runtime source; no NPU calls."""
import argparse
import hashlib
import json
from pathlib import Path

import triton

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--cache", type=Path, default=Path("/eval/triton-cache"))
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
entries = []
paths = [
    path
    for name in ("_score_pool_tile.json", "_relu_gate_inplace.json")
    for path in a.cache.rglob(name)
]
for path in sorted(paths):
    data = json.loads(path.read_text())
    entries.append(
        dict(
            path=str(path),
            source_kernel=path.stem,
            **{
                key: data.get(key)
                for key in (
                    "hash",
                    "kernel_name",
                    "name",
                    "mix_mode",
                    "workspace_size",
                    "multibuffer",
                )
            },
        )
    )
runtime = Path(triton.__file__).parent / "backends/ascend"
sources = []
for path in sorted(runtime.rglob("*.py")):
    content = path.read_text()
    lines = content.splitlines()
    hits = [
        i
        for i, line in enumerate(lines)
        if "workspace_size" in line or "workspaceSize" in line
    ]
    if hits:
        selected = sorted(
            set(j for i in hits for j in range(max(0, i - 5), min(len(lines), i + 9)))
        )
        sources.append(
            dict(
                path=str(path),
                sha256=hashlib.sha256(content.encode()).hexdigest(),
                excerpt="\n".join(f"{i+1}: {lines[i]}" for i in selected),
            )
        )
result = dict(triton=triton.__version__, kernels=entries, runtime=sources)
a.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
