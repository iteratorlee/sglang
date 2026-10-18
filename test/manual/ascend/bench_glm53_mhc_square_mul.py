"""Check FP32 square/mul bits and complete native mHC-pre; CPU or isolated NPU.

No server, model registration, distributed runtime or local GPU is imported.
The candidate is not activated by this harness outside the extracted function.
"""
import argparse
import ast
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Tuple

import torch


def load_native(path):
    tree = ast.parse(path.read_text())
    nodes = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == '_mhc_pre_torch']
    assert len(nodes) == 1
    module = ast.Module(body=nodes, type_ignores=[])
    namespace = dict(torch=torch, Tuple=Tuple)
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace['_mhc_pre_torch']


def same_bits(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    return bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


def measure(fn, device, repeats):
    values = []
    for _ in range(3):
        fn()
    for _ in range(repeats):
        if device.type == 'npu':
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record();fn();end.record();end.synchronize()
            elapsed = start.elapsed_time(end)
        else:
            begin=time.perf_counter();fn();elapsed=(time.perf_counter()-begin)*1000
        values.append(elapsed)
    return dict(n=len(values), p50_ms=statistics.median(values), mean_ms=statistics.mean(values))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-file', type=Path, required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--confirm-parent-stopped-p', action='store_true')
    ap.add_argument('--rows', type=int, nargs='+')
    ap.add_argument('--repeats', type=int, default=20)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    assert not a.output.exists()
    assert a.device == 'cpu' or a.device == 'npu:0'
    if a.device.startswith('npu'):
        assert a.confirm_parent_stopped_p
        import torch_npu  # noqa: F401
        torch.npu.set_device(0)
    dev = torch.device(a.device)
    rows = a.rows or ([32, 512] if dev.type == 'cpu' else [4097, 8192, 16384])
    fn = load_native(a.source_file)
    gen = torch.Generator(device='cpu').manual_seed(101801)
    results = []
    source = dict(path=str(a.source_file),sha256=hashlib.sha256(a.source_file.read_bytes()).hexdigest())
    try:
        # Exhaust every BF16 input bit pattern, including signed zero, infinities
        # and NaNs. Production square sees only BF16 values widened to FP32.
        exhaustive = torch.arange(-32768,32768,dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float().to(dev)
        sq, mul = exhaustive.square(), exhaustive*exhaustive
        all_bits = same_bits(sq,mul)
        finite = torch.isfinite(exhaustive)
        finite_bits = same_bits(sq[finite],mul[finite])
        results.append(dict(case='all_65536_bf16_patterns_widened_fp32',all_bits_equal=all_bits,finite_bits_equal=finite_bits))
        assert all_bits, 'FP32 square/mul differs on widened BF16 patterns'
        del exhaustive,sq,mul,finite
        for n in rows:
            x = (torch.randn((n,4,4096),generator=gen)*2).bfloat16().to(dev)
            weight=(torch.randn((24,16384),generator=gen)*0.01).to(dev)
            scale=torch.tensor([.1,.2,.3],dtype=torch.float32,device=dev)
            base=torch.linspace(-.3,.3,24,device=dev)
            kw=dict(rms_eps=1e-6,hc_pre_eps=1e-6,hc_sinkhorn_eps=1e-6,hc_post_mult_value=2.,sinkhorn_repeat=20)
            old=fn(x,weight,scale,base,**kw)
            new=fn(x,weight,scale,base,**kw,square_via_mul=True)
            parity=[same_bits(left,right) for left,right in zip(old,new)]
            assert all(parity), ('full mHC pre differs',n,parity)
            flat=x.view(n,-1).float()
            sq=flat.square();mul=flat*flat
            assert same_bits(sq,mul),('random full square differs',n)
            del old,new,sq,mul
            timing=dict(square=measure(lambda:flat.square(),dev,a.repeats),
                        mul=measure(lambda:flat*flat,dev,a.repeats),
                        full_old=measure(lambda:fn(x,weight,scale,base,**kw),dev,a.repeats),
                        full_new=measure(lambda:fn(x,weight,scale,base,**kw,square_via_mul=True),dev,a.repeats))
            row=dict(rows=n,shape=list(x.shape),full_return_bits_equal=parity,timing=timing)
            results.append(row);print(json.dumps(row),flush=True)
            del x,weight,scale,base,flat
            gc.collect()
        value=dict(passed=True,device=str(dev),source=source,results=results,whole_model_accuracy_evaluated=False)
        rc=0
    except Exception as exc:
        value=dict(passed=False,device=str(dev),source=source,results=results,error=repr(exc),whole_model_accuracy_evaluated=False)
        rc=1
    with a.output.open('x') as f:
        json.dump(value,f,indent=2)
    print(json.dumps(dict(passed=value['passed'],output=str(a.output))),flush=True)
    raise SystemExit(rc)


if __name__ == '__main__':
    main()
