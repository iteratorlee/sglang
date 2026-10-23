"""Build only pinned varlen KDA in a namespace separate from prepared_v2.

Inside the existing CANN/torch_npu container (no pip install/PYTHONPATH change):
  python scripts/npu/build_glm53_kda_ascendc.py \
    --source /path/to/01-glm53-ascend-1023 --output /path/to/kda-build
  export SGLANG_GLM53_KDA_VARLEN_ASCENDC_LIB=/path/to/kda-build/build/sgl_glm53_ascend_ops_lib.so
  export SGLANG_GLM53_KDA_VARLEN_ASCENDC=1

--prepare-only exports the exact pinned sources and namespace adaptation on a
CPU machine, without running CMake or importing torch. The original checkout
is never modified; Git blobs are read from the pinned commit, not working files.
Keep the old prepared_recurrent_v2 package, library and environment intact.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

COMMIT = "f7a4f2d7f1355b39b921962e0d768f568464e07a"
PREFIX = "sgl-glm53-ascend-ops/"
NAMESPACE = "sglang_glm53_kda_ascendc"


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError("Build output must be outside the external checkout")
    paths = subprocess.check_output(
        ["git", "-C", str(source), "ls-tree", "-r", "--name-only", COMMIT],
        text=True,
    ).splitlines()
    selected = [
        p
        for p in paths
        if p == PREFIX + "CMakeLists.txt"
        or p.startswith(PREFIX + "csrc/kda/kda_varlen_recurrent")
        or p == PREFIX + "csrc/include/utils.h"
    ]
    if len(selected) != 10:
        raise RuntimeError(f"Unexpected pinned source list: {selected}")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory (no stale object reuse)")
    manifest = dict(kernel_commit=COMMIT, namespace=NAMESPACE, sources={})
    for path in selected:
        raw = subprocess.check_output(
            ["git", "-C", str(source), "show", f"{COMMIT}:{path}"]
        )
        data = raw
        if path.endswith("/kda_varlen_recurrent.cpp"):
            for registration in ("TORCH_LIBRARY_FRAGMENT", "TORCH_LIBRARY_IMPL"):
                old = f"{registration}(sgl_glm53_ascend,".encode()
                if data.count(old) != 1:
                    raise RuntimeError(
                        f"Unexpected registration in pinned source: {registration}"
                    )
                data = data.replace(old, f"{registration}({NAMESPACE},".encode())
            marker = f"TORCH_LIBRARY_FRAGMENT({NAMESPACE}, m) {{".encode()
            data = data.replace(
                marker,
                marker
                + (
                    '\n    m.def("build_revision() -> str", []() -> std::string { '
                    f'return "{COMMIT}"; }});\n'
                ).encode(),
                1,
            )
        rel = path.removeprefix(PREFIX)
        dst = output / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        manifest["sources"][rel] = dict(
            original_sha256=hashlib.sha256(raw).hexdigest(),
            build_sha256=hashlib.sha256(data).hexdigest(),
        )
    (output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    manifest = prepare(args.source, args.output)
    root = args.output.resolve()
    if not args.prepare_only:
        subprocess.run(
            [
                "cmake",
                "-S",
                str(root),
                "-B",
                str(root / "build"),
                "-G",
                "Ninja",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DSGL_OPS=kda_varlen_recurrent",
                f"-DPython_EXECUTABLE={sys.executable}",
            ],
            check=True,
        )
        subprocess.run(
            [
                "cmake",
                "--build",
                str(root / "build"),
                "--target",
                "sgl_glm53_ascend_ops_lib",
                "--parallel",
                str(args.jobs),
            ],
            check=True,
        )
        library = root / "build/sgl_glm53_ascend_ops_lib.so"
        manifest.update(
            library=str(library),
            library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        )
        (root / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                output=str(root),
                kernel_commit=COMMIT,
                library=str(root / "build/sgl_glm53_ascend_ops_lib.so"),
                compiled=not args.prepare_only,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
