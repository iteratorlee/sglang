"""Real NPU top-k padding check; imports the frozen function via AST only."""
import ast
import argparse
import json
import math
from pathlib import Path
import torch
import torch_npu

p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
tree = ast.parse(a.source.read_text())
methods = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_pad_topk_indices"]
assert len(methods) == 1
module = ast.Module(body=methods, type_ignores=[])
ns = {"torch": torch}
exec(compile(ast.fix_missing_locations(module), str(a.source), "exec"), ns)
pad = ns["_pad_topk_indices"]
torch.npu.set_device(0)
rows = []
for dtype in (torch.int32, torch.int64):
    for count in (1, 3, 4, 16):
        for tail in ((2048,), (1, 2051)):
            cpu = torch.arange(count * math.prod(tail), dtype=dtype).reshape(count, *tail)
            for layout in ("plain", "3d_view", "format0_view"):
                x = cpu.to("npu")
                if layout != "plain":
                    x = x.view(count, 1, -1)
                    if layout == "format0_view":
                        x = torch_npu.npu_format_cast(x, 0)
                    x = x.view(cpu.shape)
                assert pad(None, x, count) is x
                padded_count = 16 if count < 16 else 32
                actual = pad(None, x, padded_count)
                assert actual.shape == (padded_count, *tail)
                expected = torch.full(actual.shape, -1, dtype=dtype)
                expected[:count].copy_(cpu)
                torch.npu.synchronize()
                assert torch.equal(actual.cpu(), expected)
                rows.append(dict(dtype=str(dtype), count=count, tail=list(tail), layout=layout,
                                 input_format=torch_npu.get_npu_format(x), output_format=torch_npu.get_npu_format(actual), passed=True))
try:
    pad(None, torch.zeros((4, 2048), dtype=torch.int32, device="npu"), 3)
except AssertionError:
    pass
else:
    raise AssertionError("Oversized input must fail")
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(dict(passed=True, cases=rows, oversized_rejected=True), indent=2) + "\n")
print(json.dumps(dict(passed=True, cases=len(rows))), flush=True)
