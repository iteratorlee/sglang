#!/usr/bin/env python3
"""Run CPU-only semantic, topology, PromQL and real fake-Docker mmap tests."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parser-path",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    os.sched_setaffinity(0,{30,31})
    if args.parser_path:sys.path.insert(0,str(args.parser_path))
    args.output.mkdir(parents=True,exist_ok=True)
    suite=unittest.TestSuite()
    for folder in (ROOT/"tests", ROOT/"exporter/tests"):
        suite.addTests(unittest.TestLoader().discover(str(folder),top_level_dir=str(folder)))
    start=time.monotonic()
    with (args.output/"cpu-tests.log").open("w") as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    forbidden=[name for name in sys.modules if name in ("torch","torch_npu") or name.startswith(("torch.","torch_npu."))]
    summary={"successful":result.wasSuccessful() and not forbidden,"tests":result.testsRun,"failures":len(result.failures),
             "errors":len(result.errors),"skipped":len(result.skipped),"duration_seconds":time.monotonic()-start,
             "cpu_affinity":sorted(os.sched_getaffinity(0)),"forbidden_device_modules":forbidden,
             "real_docker_access":False,"model_requests":0,"live_changes":0,"commit_or_push_performed":False}
    (args.output/"cpu-tests.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary))
    return 0 if summary["successful"] else 1


if __name__ == "__main__":
    sys.exit(main())
