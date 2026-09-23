"""GLM's variable-length MTP draft extend uses its dedicated KPool path."""

import ast
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Optional
from unittest.mock import Mock

import torch
import torch.nn.functional as F


def load_forward_npu(metadata):
    source = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/hardware_backend/npu/attention/glm53/kpool_indexer.py"
    )
    tree = ast.parse(source.read_text())
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendIndexerKPoolMixin"
    )
    method = copy.deepcopy(
        next(
            node
            for node in owner.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward_npu"
        )
    )
    method.body = [
        ast.Pass() if isinstance(node, ast.ImportFrom) else node for node in method.body
    ]
    namespace = {
        "Optional": Optional,
        "ForwardBatch": object,
        "torch": torch,
        "F": F,
        "_get_full_attn_metadata": lambda _: metadata,
        "index_block_table": lambda _, table: table,
        "_as_ascend_sparse_indices": lambda indices: indices,
    }
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["forward_npu"]


class EagerDraftIndexerTests(unittest.TestCase):
    def test_draft_extend_uses_multitoken_path_without_graph(self):
        metadata = NS(block_tables=torch.zeros((1, 4), dtype=torch.int32))
        draft = Mock(return_value=torch.zeros((4, 8), dtype=torch.int32))
        decode = Mock(side_effect=AssertionError("one-token decode path called"))
        indexer = NS(
            hidden_size=16,
            index_kpool_compress_gate=torch.ones((1, 16)),
            _glm53_draft_graph_steps=4,
            _project_q_key_weights=lambda x, _: (x, x, x),
            _draft_extend_topk=draft,
            _decode_topk=decode,
        )
        mode = NS(
            is_idle=lambda: False,
            is_target_verify=lambda: False,
            is_draft_extend_v2=lambda: True,
            is_extend=lambda: False,
        )
        batch = NS(forward_mode=mode)
        result = load_forward_npu(metadata)(
            indexer,
            x=torch.ones((4, 16)),
            q_lora=torch.ones((4, 16)),
            positions=torch.arange(4),
            forward_batch=batch,
            layer_id=0,
        )
        self.assertEqual(tuple(result.shape), (4, 8))
        draft.assert_called_once()
        decode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
