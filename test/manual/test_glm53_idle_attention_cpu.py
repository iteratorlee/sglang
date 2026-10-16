"""CPU scope and decoder-flow tests; never initialize an accelerator."""

import ast
from contextlib import nullcontext
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
NPU = ROOT / "python/sglang/srt/hardware_backend/npu/moe"
FLAG = "SGLANG_GLM53_NORMAL_SKIP_IDLE_ATTN"


class IdleAttentionTests(unittest.TestCase):
    def setUp(self):
        self.profile = True
        self.parallel = NS(attn_dp_size=2, attn_tp_size=8, attn_dp_rank=0)
        collective = ModuleType("collective")
        collective.normal_hccl_skip_idle_dp_enabled = lambda: self.profile
        collective.normal_hccl_idle_dp_scope = lambda *args: nullcontext()
        modes = ModuleType("modes")
        modes.ForwardMode = NS(EXTEND=1)
        runtime = ModuleType("runtime")
        runtime.get_parallel = lambda: self.parallel
        spec = importlib.util.spec_from_file_location("idle_attention", NPU / "glm53_idle_attention.py")
        self.helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.helper)
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {
            "sglang.srt.hardware_backend.npu.moe.glm53_collectives": collective,
            "sglang.srt.hardware_backend.npu.moe.glm53_idle_attention": self.helper,
            "sglang.srt.model_executor.forward_batch_info": modes,
            "sglang.srt.runtime_context": runtime,
        }).start()
        patch.dict(os.environ, {FLAG: "1"}).start()
        self.batch = NS(forward_mode=1, dp_padding_mode=NS(is_max_len=lambda: True),
                        original_global_num_tokens_cpu=[0, 8192],
                        global_num_tokens_cpu=[8192, 8192])

    def skip(self, **kwargs):
        return self.helper.skip_idle_attention(self.batch, is_nextn=kwargs.get("is_nextn", False),
                                               local_rows=kwargs.get("local_rows", 8192))

    def test_idle_domain_only_both_directions(self):
        for original in ([0, 8192], [8192, 0]):
            self.batch.original_global_num_tokens_cpu = original
            for rank in (0, 1):
                self.parallel.attn_dp_rank = rank
                self.assertEqual(self.skip(), original[rank] == 0)

    def test_default_and_profile_exclusions(self):
        with patch.dict(os.environ, {FLAG: "0"}):
            self.assertFalse(self.skip())
        self.profile = False
        self.assertFalse(self.skip())
        self.profile = True
        self.assertFalse(self.skip(is_nextn=True))
        for mode in (0, 2, 3, 4, 5):
            self.batch.forward_mode = mode
            self.assertFalse(self.skip())

    def test_malformed_and_active_counts_are_never_skipped(self):
        for original in (None, [], [0], [0, -1], [0, True], [0, 8193], [0, 0], [8, 8192]):
            self.batch.original_global_num_tokens_cpu = original
            self.assertFalse(self.skip())

    def test_padding_shape_topology_and_bound(self):
        self.assertFalse(self.skip(local_rows=4096))
        self.batch.dp_padding_mode = NS(is_max_len=lambda: False)
        self.assertFalse(self.skip())
        self.batch.dp_padding_mode = NS(is_max_len=lambda: True)
        self.batch.global_num_tokens_cpu = [8192, 4096]
        self.assertFalse(self.skip())
        self.batch.global_num_tokens_cpu = [16384, 16384]
        self.assertTrue(self.skip(local_rows=16384))
        self.batch.global_num_tokens_cpu = [32768, 32768]
        self.assertFalse(self.skip(local_rows=32768))
        self.batch.global_num_tokens_cpu = [8192, 8192]
        self.parallel.attn_tp_size = 2
        self.assertFalse(self.skip())

    def test_decoder_retains_mlp_and_postprocess(self):
        source = ROOT / "python/sglang/srt/models/glm5_next.py"
        tree = ast.parse(source.read_text())
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "Glm5NextDecoderLayer")
        fn = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "forward")
        fn.returns = None
        for arg in fn.args.args:
            arg.annotation = None
        namespace = dict(torch=torch, nullcontext=nullcontext, Glm5NextMLP=type("Dense", (), {}),
                         Glm5NextMoE=type("MoE", (), {}),
                         get_attn_tp_context=lambda: NS(clear_attn_inputs=lambda: None),
                         get_forward=lambda: NS(scoped=lambda **kwargs: nullcontext()))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(source), "exec"), namespace)
        for nextn, active in ((False, False), (False, True), (True, False)):
            calls = []
            self.parallel.attn_dp_rank = int(active)
            def attention(**kwargs):
                calls.append("attention")
                return kwargs["hidden_states"] + 3
            def mlp(x, *args):
                calls.append("experts")
                return x + 2
            def prepare(x, residual, batch):
                calls.append("prepare")
                return x, residual
            def finish(x, residual, batch):
                calls.append("postprocess")
                return x, residual
            comm = NS(prepare_attn=prepare, prepare_mlp=prepare, postprocess_layer=finish,
                      should_fuse_mlp_allreduce_with_next_layer=lambda b: False,
                      should_use_reduce_scatter=lambda b: False)
            layer = NS(_normal_skip_idle_dp=True, is_nextn=nextn, self_attn=attention,
                       mlp=mlp, layer_communicator=comm, layer_scatter_modes=None)
            x = torch.ones(8192, 4)
            output, _, _ = namespace["forward"](layer, None, x, self.batch, None)
            skipped = not nextn and not active
            self.assertEqual("attention" in calls, not skipped)
            self.assertEqual(calls.count("experts"), 1)
            self.assertEqual(calls.count("prepare"), 2)
            self.assertEqual(calls.count("postprocess"), 1)
            torch.testing.assert_close(output, torch.full_like(x, 2 if skipped else 6), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
