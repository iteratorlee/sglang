"""Numerical and graph replay checks for the native GLM-5.3 Ascend port.

Run in the Ascend image after sourcing its CANN environment.
"""

import unittest
import os
from unittest.mock import patch
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import torch_npu
import sglang.srt.layers.quantization  # initialize the native quantization registry

from sglang.kernels.ops.layernorm.mhc import (
    _mhc_post_torch,
    _mhc_pre_torch,
    hc_post,
    hc_pre,
)
from sglang.srt.hardware_backend.npu.quantization.w8a8_clamped_moe import (
    NPUW8A8ClampedMoEMethod,
)


class TestGlm53NativeOps(unittest.TestCase):
    def setUp(self):
        torch.npu.set_device(0)
        torch.manual_seed(20260914)

    def test_compact_pool_storage_and_dynamic_graph_addresses(self):
        from sglang.srt.hardware_backend.npu.attention.glm53.compact_index import (
            make_layout,
        )
        from sglang.srt.hardware_backend.npu.attention.glm53.compact_index_npu import (
            request_block_table,
        )
        from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool

        layout = make_layout(2, 1024, 8192, draft_tokens=4)
        pool = NPUMLATokenToKVPool(
            size=8192,
            page_size=64,
            dtype=torch.bfloat16,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            layer_num=3,
            device="npu",
            enable_memory_saver=False,
            index_head_dim=128,
            index_layout=layout,
            share_zero_rope=True,
        )
        self.assertEqual(pool.index_k_buffer.shape[1], layout.pages)
        self.assertEqual(pool.v_buffer.stride(0), 0)
        self.assertTrue(pool.get_value_buffer(2).is_contiguous())
        self.assertEqual(
            pool.get_value_buffer(0).data_ptr(), pool.get_value_buffer(2).data_ptr()
        )
        expected_bytes = (
            3 * 129 * 64 * 512 + 129 * 64 * 64 + 3 * layout.pages * 64 * 128
        ) * 2
        self.assertEqual(pool.get_kv_size_bytes(), expected_bytes)
        loc = torch.tensor([64, 130], device="npu", dtype=torch.int32)
        values = torch.randn(2, 1, 512, device="npu", dtype=torch.bfloat16)
        pool.set_kv_buffer(SimpleNamespace(layer_id=1), loc, values, values[..., :0])
        torch.testing.assert_close(
            pool.get_key_buffer(1).flatten(0, 1)[loc.long()], values
        )
        self.assertEqual(torch.count_nonzero(pool.v_buffer).item(), 0)

        ids = torch.tensor([2, 1, 0], device="npu", dtype=torch.int32)
        original = torch.zeros(3, 17, device="npu", dtype=torch.int32)
        for _ in range(3):
            output = request_block_table(ids, original, layout)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = request_block_table(ids, original, layout)
        for slots in ([2, 1, 0], [0, 2, 1], [1, 0, 2]):
            ids.copy_(torch.tensor(slots, device="npu", dtype=torch.int32))
            graph.replay()
            expected = torch.tensor(
                [
                    [
                        (
                            1
                            + (r - 1) * layout.pages_per_request
                            + min(c // 4, layout.pages_per_request - 1)
                            if r
                            else 0
                        )
                        for c in range(17)
                    ]
                    for r in slots
                ],
                dtype=torch.int32,
            )
            torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)

    def test_kda_glm_and_kimi_gate_contract(self):
        from sglang.srt.hardware_backend.npu.attention.ascend_kda_backend import (
            AscendKDAAttnBackend,
        )

        backend = object.__new__(AscendKDAAttnBackend)
        for lower_bound in (None, -5.0):
            layer = SimpleNamespace(
                A_log=torch.randn(1, 1, 4, 1, device="npu"),
                head_k_dim=128,
                dt_bias=torch.randn(512, device="npu"),
                lower_bound=lower_bound,
            )
            a = torch.randn(1, 7, 512, device="npu", dtype=torch.bfloat16)
            b = torch.randn(1, 7, 4, device="npu", dtype=torch.bfloat16)
            g, beta, _, _ = backend._prepare_extend_gate_inputs(layer, a, b)
            activated = a.float().reshape(1, 7, 4, 128) + layer.dt_bias.reshape(4, 128)
            expected = (
                -layer.A_log.exp() * F.softplus(activated)
                if lower_bound is None
                else lower_bound * torch.sigmoid(layer.A_log.exp() * activated)
            )
            torch.testing.assert_close(g, expected, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(beta, b.float().sigmoid(), atol=0, rtol=0)
            kimi_g, kimi_beta, _, _ = backend._prepare_extend_gate_inputs(
                layer, a.reshape(1, 7, 4, 128), beta
            )
            torch.testing.assert_close(kimi_g, g, atol=0, rtol=0)
            torch.testing.assert_close(kimi_beta, beta, atol=0, rtol=0)

    def test_glm_kda_state_layout_and_snapshots(self):
        from sglang.srt.hardware_backend.npu.attention.glm53.kda_recurrent_npu import (
            glm_kda_varlen_recurrent_npu,
        )

        batch, steps, heads, width = 2, 3, 4, 128
        q, k, v, a = [
            torch.randn(
                1, batch * steps, heads, width, dtype=torch.bfloat16, device="npu"
            )
            for _ in range(4)
        ]
        b = torch.randn(1, batch * steps, heads, dtype=torch.bfloat16, device="npu")
        alog = torch.randn(1, 1, heads, 1, device="npu")
        bias = torch.randn(heads * width, device="npu")
        starts = torch.tensor([0, 3, 6], device="npu", dtype=torch.int32)
        ids = torch.tensor([2, 1], device="npu", dtype=torch.int32)
        seed = torch.randn(4, heads, width, width, device="npu") * 0.1
        expected_states, expected_outputs = [], []
        for row, slot in enumerate((2, 1)):
            h = seed[slot].cpu().float()
            snapshots, outputs = [], []
            for t in range(row * steps, (row + 1) * steps):
                qi, ki, vi, ai = [x[0, t].cpu().float() for x in (q, k, v, a)]
                qi = (
                    qi
                    * torch.rsqrt(qi.square().sum(-1, keepdim=True) + 1e-6)
                    / width**0.5
                )
                ki = ki * torch.rsqrt(ki.square().sum(-1, keepdim=True) + 1e-6)
                gate = -5.0 * torch.sigmoid(
                    alog.cpu().reshape(heads, 1).exp()
                    * (ai + bias.cpu().reshape(heads, width))
                )
                h = h * gate.exp().unsqueeze(-1)
                update = (vi - (h * ki.unsqueeze(-1)).sum(-2)) * b[
                    0, t
                ].cpu().float().sigmoid().unsqueeze(-1)
                h = h + ki.unsqueeze(-1) * update.unsqueeze(-2)
                snapshots.append(h.clone())
                outputs.append((h * qi.unsqueeze(-1)).sum(-2))
            expected_states.append(torch.stack(snapshots))
            expected_outputs.extend(outputs)
        expected_states = torch.stack(expected_states)
        expected_outputs = torch.stack(expected_outputs).unsqueeze(0).bfloat16()
        for transpose_storage in (False, True):
            for verify in (False, True):
                state = (
                    seed.transpose(-1, -2).contiguous().transpose(-1, -2)
                    if transpose_storage
                    else seed.clone()
                )
                snapshots = (
                    torch.zeros(
                        batch, steps, heads, width, width, device="npu"
                    ).transpose(-1, -2)
                    if verify
                    else None
                )
                result = glm_kda_varlen_recurrent_npu(
                    q=q,
                    k=k,
                    v=v,
                    a=a,
                    b=b,
                    A_log=alog,
                    dt_bias=bias,
                    initial_state_source=state,
                    initial_state_indices=ids,
                    cu_seqlens=starts,
                    lower_bound=-5.0,
                    intermediate_state=snapshots,
                )
                torch.testing.assert_close(
                    result.cpu(), expected_outputs, atol=0.001, rtol=0.01
                )
                if verify:
                    torch.testing.assert_close(
                        snapshots.cpu(), expected_states, atol=1e-5, rtol=1e-4
                    )
                    torch.testing.assert_close(state, seed, atol=0, rtol=0)
                else:
                    torch.testing.assert_close(
                        state[ids.long()].cpu(),
                        expected_states[:, -1],
                        atol=1e-5,
                        rtol=1e-4,
                    )

    def test_kpool_accepted_state_graph_replay(self):
        from sglang.srt.hardware_backend.npu.attention.glm53.accepted_state import (
            commit_kpool_tails,
        )

        idx = SimpleNamespace()
        for suffix, dtype in (("k", torch.bfloat16), ("score", torch.float32)):
            setattr(
                idx,
                "_kpool_tail_" + suffix,
                torch.zeros(6, 4, 128, device="npu", dtype=dtype),
            )
            setattr(
                idx,
                "_kpool_mtp_tail_" + suffix,
                torch.randn(2, 4, 4, 128, device="npu", dtype=dtype),
            )
        model = SimpleNamespace(
            model=SimpleNamespace(
                layers=[SimpleNamespace(self_attn=SimpleNamespace(indexer=idx))]
            )
        )
        backend = SimpleNamespace()
        for requests, steps in (([3, 1], [0, 2]), ([2, 4], [3, 1])):
            req = torch.tensor(requests, device="npu", dtype=torch.int32)
            accepted = torch.tensor(steps, device="npu", dtype=torch.int32)
            commit_kpool_tails(backend, model, accepted, req)
            for suffix in ("k", "score"):
                for row, (slot, step) in enumerate(zip(requests, steps)):
                    torch.testing.assert_close(
                        getattr(idx, "_kpool_tail_" + suffix)[slot],
                        getattr(idx, "_kpool_mtp_tail_" + suffix)[row, step],
                        rtol=0.0,
                        atol=0.0,
                    )

    def test_kda_prepared_prefill_matches_recurrence(self):
        from sglang.srt.hardware_backend.npu.attention.glm53.kda_recurrent_npu import (
            glm_kda_varlen_recurrent_npu,
        )

        q, k, v, a = [
            torch.randn(1, 128, 4, 128, dtype=torch.bfloat16, device="npu")
            for _ in range(4)
        ]
        b = torch.randn(1, 128, 4, device="npu", dtype=torch.bfloat16)
        alog = torch.randn(1, 1, 4, 1, device="npu")
        bias = torch.randn(512, device="npu")
        starts = torch.tensor([0, 128], device="npu", dtype=torch.int32)
        ids = torch.tensor([2], device="npu", dtype=torch.int32)
        seed = torch.randn(4, 4, 128, 128, device="npu") * 0.1
        reference_state = seed.clone().transpose(-1, -2)
        prepared_state = seed.clone().transpose(-1, -2)
        kwargs = dict(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            A_log=alog,
            dt_bias=bias,
            initial_state_indices=ids,
            cu_seqlens=starts,
            lower_bound=-5.0,
        )
        reference = glm_kda_varlen_recurrent_npu(
            **kwargs, initial_state_source=reference_state
        )
        with patch.dict(os.environ, {"SGLANG_GLM53_KDA_PREFILL_PREPARE": "1"}):
            actual = glm_kda_varlen_recurrent_npu(
                **kwargs, initial_state_source=prepared_state, prefill=True
            )
        torch.testing.assert_close(actual, reference, rtol=0.01, atol=0.001)
        torch.testing.assert_close(
            prepared_state, reference_state, rtol=1e-4, atol=1e-5
        )

    def test_mhc_fused_norm_and_post(self):
        for rows in (1, 4, 16, 129):
            x = torch.randn(rows, 16384, device="npu", dtype=torch.bfloat16)
            fn = torch.randn(24, 16384, device="npu") * 0.002
            scale = torch.tensor([0.1, 0.2, 0.3], device="npu")
            base = torch.randn(24, device="npu") * 0.1
            weight = torch.randn(4096, device="npu", dtype=torch.bfloat16)
            post, comb, mixed = _mhc_pre_torch(
                x.reshape(rows, 4, 4096),
                fn,
                scale,
                base,
                1e-6,
                1e-6,
                1e-6,
                2.0,
                20,
            )
            expected = torch.ops.npu.npu_rms_norm(mixed, weight, 1e-6)[0]
            actual, actual_comb, actual_post, fused = hc_pre(
                x,
                fn,
                scale,
                base,
                4,
                1e-6,
                1e-6,
                20,
                out_norm_weight=weight,
                out_norm_eps=1e-6,
            )
            self.assertTrue(fused)
            torch.testing.assert_close(
                actual_comb, comb.reshape(rows, 16), atol=2e-6, rtol=2e-5
            )
            torch.testing.assert_close(
                actual_post, post.reshape(rows, 4), atol=2e-6, rtol=2e-5
            )
            torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.02)
            actual_out = hc_post(actual, x, actual_post, actual_comb, 4)
            expected_out = _mhc_post_torch(
                actual,
                x.reshape(rows, 4, 4096),
                actual_post.reshape(rows, 4, 1),
                actual_comb.reshape(rows, 4, 4),
            )
            torch.testing.assert_close(
                actual_out, expected_out.reshape(rows, -1), atol=0.015625, rtol=0.01
            )

    def test_w8a8_clamp_empty_experts_and_padding(self):
        for capacity in (8, 512):
            experts, hidden, intermediate = 4, 256, 128
            layer = torch.nn.Module()
            originals = {}
            for prefix, channels, inputs in (
                ("w13", 2 * intermediate, hidden),
                ("w2", hidden, intermediate),
            ):
                w = torch.randint(
                    -32, 32, (experts, channels, inputs), device="npu", dtype=torch.int8
                )
                s = torch.rand(experts, channels, 1, device="npu") * 0.008 + 0.012
                originals[prefix] = (w.clone(), s.squeeze(-1).clone())
                layer.register_parameter(
                    prefix + "_weight", torch.nn.Parameter(w, requires_grad=False)
                )
                layer.register_parameter(
                    prefix + "_weight_scale", torch.nn.Parameter(s, requires_grad=False)
                )
                layer.register_parameter(
                    prefix + "_weight_offset",
                    torch.nn.Parameter(torch.zeros_like(s), requires_grad=False),
                )
            kernel = NPUW8A8ClampedMoEMethod(3.0)
            for prefix in ("w13", "w2"):
                kernel.process_weights_after_loading(layer, prefix)
                self.assertEqual(getattr(layer, prefix + "_weight").dtype, torch.int8)
                self.assertEqual(
                    getattr(layer, prefix + "_weight_scale").dtype, torch.float32
                )
            qi = SimpleNamespace(**dict(layer.named_parameters()))
            x = torch.randn(capacity, hidden, device="npu", dtype=torch.bfloat16) * 3
            xq, xs = torch.ops.npu.npu_dynamic_quant(x)
            counts = torch.tensor([0, 3, 0, 5], device="npu", dtype=torch.int64)
            yq, ys = kernel.apply_fused_gmm1_swiglu(qi, xq, counts, xs, 1)
            y = kernel.apply(qi, yq, counts, ys, torch.bfloat16, "w2", 1)
            row = 0
            for e, count in enumerate((0, 3, 0, 5)):
                if not count:
                    continue
                a = xq[row : row + count].float() @ originals["w13"][0][e].float().T
                a = a * xs[row : row + count, None] * originals["w13"][1][e]
                gate, up = a.chunk(2, -1)
                expected_act = F.silu(gate.clamp(max=3.0)) * up.clamp(-3.0, 3.0)
                reconstructed = (
                    yq[row : row + count].float() * ys[row : row + count, None]
                )
                torch.testing.assert_close(
                    reconstructed, expected_act, atol=0.08, rtol=0.03
                )
                expected = (
                    yq[row : row + count].float() @ originals["w2"][0][e].float().T
                )
                expected = (
                    expected * ys[row : row + count, None] * originals["w2"][1][e]
                )
                torch.testing.assert_close(
                    y[row : row + count], expected.bfloat16(), atol=0.0, rtol=0.0
                )
                row += count
            self.assertTrue(torch.count_nonzero(y[8:]).item() == 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
