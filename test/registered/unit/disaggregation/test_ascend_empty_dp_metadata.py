"""An idle attention-DP domain must survive a speculative verify step."""

import ast
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch


def load_init_forward_metadata():
    source = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py"
    )
    tree = ast.parse(source.read_text())
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendAttnBackend"
    )
    method = copy.deepcopy(
        next(
            node
            for node in owner.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "init_forward_metadata"
        )
    )
    method.decorator_list = []
    namespace = {
        "ForwardBatch": object,
        "ForwardMetadata": NS,
        "np": np,
        "torch": torch,
    }
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["init_forward_metadata"]


class EmptyDPMetadataTests(unittest.TestCase):
    def test_idle_domain_target_verify(self):
        forward_mode = NS(
            is_target_verify=lambda: True,
            is_draft_extend_v2=lambda: False,
            is_decode_or_idle=lambda: False,
            is_extend=lambda: False,
        )
        empty = torch.empty(0, dtype=torch.int32)
        batch = NS(
            seq_lens=empty,
            seq_lens_cpu=empty,
            req_pool_indices=empty,
            forward_mode=forward_mode,
            spec_algorithm=None,
            spec_info=NS(draft_token_num=4),
            extend_seq_lens=None,
            out_cache_loc=None,
        )
        backend = NS(
            req_to_token_pool=NS(req_to_token=torch.zeros((1, 64), dtype=torch.int32)),
            page_size=64,
            use_mla=False,
            is_hybrid_swa=False,
            use_sliding_window_kv_pool=False,
            device="cpu",
        )
        load_init_forward_metadata()(backend, batch)
        self.assertEqual(tuple(backend.forward_metadata.block_tables.shape), (0, 0))
        self.assertEqual(backend.forward_metadata.seq_lens_cpu_int.numel(), 0)
        self.assertEqual(backend.forward_metadata.actual_seq_lengths_q.numel(), 0)


if __name__ == "__main__":
    unittest.main()
