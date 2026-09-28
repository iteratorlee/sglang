"""Real-910B regression for GLM KDA speculative state and verify commit views.

Run from a writable working directory with the SGLang source on PYTHONPATH.
The target model is not needed. This exercises the same recurrent kernel and
speculative state mover as serving.
"""

import json
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.mamba.mamba_state_update_triton import move_intermediate_cache_kda

import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.hardware_backend.npu.attention.ascend_kda_backend import (
    AscendKDAAttnBackend,
)
from sglang.srt.hardware_backend.npu.attention.glm53.kda_recurrent_npu import (
    glm_kda_varlen_recurrent_npu,
)


def max_error(left, right):
    return float((left.detach().float().cpu() - right.detach().float().cpu()).abs().max())


def main():
    torch.npu.set_device(0)
    torch.manual_seed(17)
    dev, heads, dim = "npu:0", 4, 128
    layer = SimpleNamespace(
        num_v_heads=heads,
        head_k_dim=dim,
        A_log=torch.randn((1, 1, heads, 1), dtype=torch.float32, device=dev),
        dt_bias=torch.randn((1, 1, heads, dim), dtype=torch.float32, device=dev),
        lower_bound=-10.0,
    )
    indices = torch.tensor([1], dtype=torch.int32, device=dev)

    def inputs(tokens):
        q, k, v, a = (
            torch.randn((1, tokens, heads, dim), dtype=torch.bfloat16, device=dev)
            for _ in range(4)
        )
        b = torch.randn((1, tokens, heads), dtype=torch.bfloat16, device=dev)
        starts = torch.tensor([0, tokens], dtype=torch.int32, device=dev)
        return q, k, v, a, b, starts

    def run(pool, tensors, *, intermediate=None, prefill=False):
        q, k, v, a, b, starts = tensors
        return AscendKDAAttnBackend._glm_recurrent(
            None, layer, q, k, v, a, b, pool, indices, starts,
            intermediate=intermediate, prefill=prefill,
        )

    # MTP's persistent pool is already a transpose view; the backend must
    # present exactly the same [K,V] stride to the kernel as the non-MTP pool.
    prefill_input = inputs(26)
    baseline = torch.zeros((2, heads, dim, dim), dtype=torch.float32, device=dev)
    speculative = baseline.clone().transpose(-1, -2)
    prefill_errors = []
    for _ in range(2):
        ref = run(baseline, prefill_input, prefill=True)
        got = run(speculative, prefill_input, prefill=True)
        prefill_errors.append(max_error(ref, got))
    state_error = max_error(baseline.transpose(-1, -2), speculative)

    # A verify writes its accepted states to scratch. Commit must use the
    # physical destination view, otherwise the next recurrent call reads a
    # transposed matrix although the immediate verify output may look right.
    verify_input = inputs(4)
    baseline.zero_()
    speculative.zero_()
    ref = run(baseline, verify_input)
    scratch = torch.zeros((1, 1, 4, heads, dim, dim), dtype=torch.float32, device=dev)
    got = run(speculative, verify_input, intermediate=scratch[0])
    move_intermediate_cache_kda(
        speculative.unsqueeze(0).transpose(-1, -2), scratch,
        torch.tensor([1], dtype=torch.int32, device=dev),
        torch.tensor([0], dtype=torch.int32, device=dev),
        torch.tensor([3], dtype=torch.int32, device=dev),
        h_block_size=1,
    )
    verify_error = max_error(ref, got)
    commit_error = max_error(baseline.transpose(-1, -2), speculative)
    result = {
        "prefill_output_max_abs": prefill_errors,
        "prefill_state_max_abs": state_error,
        "verify_output_max_abs": verify_error,
        "committed_state_max_abs": commit_error,
    }
    print(json.dumps(result), flush=True)
    assert all(v == 0 for v in prefill_errors), result
    assert state_error == verify_error == commit_error == 0, result


if __name__ == "__main__":
    main()
