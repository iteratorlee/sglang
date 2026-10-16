"""Standalone single-NPU ordered-combine check; no model or HCCL group.

Run on an idle test device, outside performance sampling:
  python test/manual/ascend/test_glm53_ordered_combine_npu.py \
      --device npu:0 --rows 0 16 16384

Use --rows 131056 explicitly for the maximum EP16 geometry (about 2 GiB per
FP32 output). The default largest output is 256 MiB. CPU tests import the
reference functions only; torch_npu is imported exclusively by main().
"""

import argparse
import importlib.util
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
MOE = ROOT / "python/sglang/srt/hardware_backend/npu/moe"


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_inputs(rows, width, case):
    """Sparse, stable expert-order contributions including ties and eight slots."""
    targets = sorted({0, 1, rows // 2 - 1, rows // 2, rows - 1})
    entries = []
    for row in targets:
        if not 0 <= row < rows or case == "empty":
            continue
        if case == "idle_first" and row < rows // 2:
            continue
        if case == "idle_second" and row >= rows // 2:
            continue
        entries.extend(((slot * 5 + row) % 18, row) for slot in range(8))
    entries.sort(key=lambda entry: entry[0])  # Python sort is stable.
    source_rows = torch.tensor([row for _, row in entries], dtype=torch.int64)
    values = torch.arange(len(entries) * width, dtype=torch.float32).view(-1, width)
    y = ((values % 31 - 15) / 8).to(torch.bfloat16)
    weights = 0.125 + (torch.arange(len(entries), dtype=torch.float32) + 1) * 2**-20
    # These detect FP32 reassociation, premature BF16 accumulation, and loss
    # of FP32 gate mantissa bits. Keep the actual expert order fixed.
    for row in targets:
        locations = (source_rows == row).nonzero().flatten()
        if locations.numel() >= 5:
            y[locations, :3] = 0
            for index, value in zip(locations[:3], (2**24, 1, -(2**24))):
                y[index, 0] = value
                weights[index] = 1
            # FP32 add must retain 256 + 1 - 256; BF16 accumulation loses 1.
            y[locations[:3], 2] = torch.tensor([256, 1, -256], dtype=torch.bfloat16)
            # Separately rounded product gives 2**-23; fused multiply-add
            # gives a different result for the last two contributions.
            y[locations[3], 1], weights[locations[3]] = -1.0078125, 1
            y[locations[4], 1] = 1.0078125
            weights[locations[4]] = torch.nextafter(
                torch.tensor(1.0), torch.tensor(2.0)
            )
    return y, weights, source_rows


def sequential_reference(y, weights, source_rows, rows):
    """Independent per-contribution FP32 multiply, round, then FP32 add."""
    out = torch.zeros(rows, y.shape[1], dtype=torch.float32)
    for index, row in enumerate(source_rows.tolist()):
        product = y[index].float() * weights[index]
        out[row] = out[row] + product
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--rows", type=int, nargs="+", default=[0, 16, 16384])
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["mixed", "empty", "idle_first", "idle_second"],
        choices=["mixed", "empty", "idle_first", "idle_second"],
    )
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()
    if args.repeat < 1 or any(
        rows < 0 or rows > 131070 or rows % 16 for rows in args.rows
    ):
        parser.error("rows must be EP16 multiples in [0, 131070], repeat >= 1")

    import torch_npu  # noqa: F401; explicitly loaded only for the NPU entry point

    torch.npu.set_device(args.device)
    collectives = load_source("glm53_collectives_test", MOE / "glm53_collectives.py")
    kernel = load_source("glm53_ordered_test", MOE / "glm53_ordered_combine.py")
    for rows in args.rows:
        for case in args.cases:
            y, weights, source_rows = make_inputs(rows, 4096, case)
            expected = sequential_reference(y, weights, source_rows, rows)
            indices = collectives._normal_combine_ordered_indices(source_rows, rows, 8)
            dy, dw, di = [value.to(args.device) for value in (y, weights, indices)]
            for repeat in range(args.repeat):
                out = kernel.ordered_combine(dy, dw, di, rows, 4096)
                torch.npu.synchronize()
                actual = out.cpu()
                assert out.dtype == torch.float32
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                del actual, out
            print(
                json.dumps(
                    dict(
                        rows=rows,
                        case=case,
                        received=len(source_rows),
                        repeats=args.repeat,
                        fp32_exact=True,
                    )
                ),
                flush=True,
            )
            del expected, dy, dw, di


if __name__ == "__main__":
    main()
