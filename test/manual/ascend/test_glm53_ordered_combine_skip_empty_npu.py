"""Remote-only NPU numerical and timing comparison for empty-tile pruning.

Run outside serving measurements. --module-dir allows immutable isolated
copies of these files, without changing a running model's source mount.
"""
import argparse
import json
from pathlib import Path
import statistics

import torch

from test_glm53_ordered_combine_npu import (
    MOE, load_source, make_inputs, sequential_reference,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--module-dir', type=Path, default=MOE)
    ap.add_argument('--device', default='npu:0')
    ap.add_argument('--rows', type=int, nargs='+', default=[1024, 16384, 32768])
    ap.add_argument('--repeat', type=int, default=20)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    import torch_npu  # noqa: F401
    torch.npu.set_device(a.device)
    module = load_source('empty_tile_kernel', a.module_dir / 'glm53_ordered_combine.py')
    collectives = load_source('empty_tile_indices', a.module_dir / 'glm53_collectives.py')
    checks = []
    for rows in (1, 2, 3, 17, 256):
        for case in ('mixed', 'empty', 'idle_first', 'idle_second'):
            y, w, source_rows = make_inputs(rows, 4096, case)
            index = collectives._normal_combine_ordered_indices(source_rows, rows, 8)
            expected = sequential_reference(y, w, source_rows, rows)
            dy, dw, di = [t.to(a.device) for t in (y, w, index)]
            actual = module.ordered_combine(dy, dw, di, rows, 4096, skip_empty=True)
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
            checks.append(dict(rows=rows, case=case, fp32_exact=True))
            del actual, dy, dw, di, expected
    trials = []
    gen = torch.Generator().manual_seed(20261016)
    for rows in a.rows:
        for active_fraction in (0.5, 1.0):
            for case in (('random', 'concentrated') if rows <= 2048 else ('random',)):
                active = int(rows * active_fraction)
                ids = torch.randint(288, (active, 8), generator=gen)
                if case == 'concentrated':
                    ids = ids.remainder(18)
                flat = ids.flatten()
                selected = (flat < 18).nonzero().flatten()
                selected = selected[torch.argsort(flat[selected], stable=True)]
                source_rows = selected // 8
                index = collectives._normal_combine_ordered_indices(source_rows, rows, 8)
                y = torch.randn(len(selected), 4096, generator=gen).to(torch.bfloat16)
                w = torch.rand(len(selected), generator=gen)
                dy, dw, di = [t.to(a.device) for t in (y, w, index)]
                old = module.ordered_combine(dy, dw, di, rows, 4096)
                new = module.ordered_combine(dy, dw, di, rows, 4096, skip_empty=True)
                torch.testing.assert_close(new.cpu(), old.cpu(), rtol=0, atol=0)
                del old, new
                samples = {False: [], True: []}
                for repeat in range(a.repeat + 3):
                    # Alternate order so the serving process' idle background
                    # work cannot systematically favor the same variant.
                    for enabled in ((False, True) if repeat % 2 else (True, False)):
                        start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
                        start.record()
                        out = module.ordered_combine(dy, dw, di, rows, 4096, skip_empty=enabled)
                        end.record()
                        end.synchronize()
                        if repeat >= 3:
                            samples[enabled].append(start.elapsed_time(end))
                        del out
                result = dict(rows=rows, active_fraction=active_fraction, case=case,
                              received=len(selected), fp32_exact=True,
                              baseline_median_ms=statistics.median(samples[False]),
                              candidate_median_ms=statistics.median(samples[True]),
                              baseline_samples_ms=samples[False], candidate_samples_ms=samples[True])
                trials.append(result)
                print(json.dumps(result), flush=True)
                del dy, dw, di
    a.output.write_text(json.dumps({'numerical': checks, 'trials': trials}, indent=2) + '\n')


if __name__ == '__main__':
    main()
