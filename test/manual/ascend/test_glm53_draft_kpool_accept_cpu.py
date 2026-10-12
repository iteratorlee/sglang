"""CPU regression for the real GLM53 draft KPool acceptance routing.

Run with a local PyTorch interpreter:
    python -B test/manual/ascend/test_glm53_draft_kpool_accept_cpu.py -v

Extract production functions by AST, without importing SGLang/NPU/Triton.
Mock _decode_topk and the scatter kernel launch to inspect their arguments;
no KPool, scatter, or attention kernel is reimplemented here. The pinned git
baseline is the audited D version. CPU tracing checks tensor dependencies,
not CANN graph replay, persistent NPU state, or model accuracy.
"""

import ast
import copy
import enum
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[3]
INDEX = "python/sglang/srt/hardware_backend/npu/attention/glm53/kpool_indexer.py"
GRAPH_OPS = ROOT / INDEX.replace("kpool_indexer.py", "graph_ops.py")
FORWARD_BATCH = ROOT / "python/sglang/srt/model_executor/forward_batch_info.py"
BASELINE = "ee31fcc955791d07431848ceea20e8083dd76435"
OWNER = "AscendIndexerKPoolMixin"
SOURCE = (ROOT / INDEX).read_text()
BASE_SOURCE = subprocess.check_output(
    ["git", "-C", str(ROOT), "show", f"{BASELINE}:{INDEX}"], text=True
)
NS = types.SimpleNamespace


def extract(source, name, namespace, *, owner=None, filename=INDEX):
    nodes = ast.parse(source).body
    if owner is not None:
        nodes = next(
            n for n in nodes if isinstance(n, ast.ClassDef) and n.name == owner
        ).body
    found = [
        n
        for n in nodes
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name
    ]
    if len(found) != 1:
        raise AssertionError((filename, owner, name, len(found)))
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            found[0],
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(filename), "exec"), namespace)
    return namespace[name]


MODE = extract(
    FORWARD_BATCH.read_text(),
    "ForwardMode",
    {"IntEnum": enum.IntEnum, "auto": enum.auto},
    filename=FORWARD_BATCH,
)


class Calls:
    """Invoke the real routing code, stopping at compute/launch boundaries."""

    def __init__(self, source=SOURCE):
        self.launch = Mock()
        kernel = MagicMock()
        kernel.__getitem__.return_value = self.launch
        scatter = extract(
            GRAPH_OPS.read_text(),
            "scatter_rows_",
            {
                "_scatter_rows_kernel": kernel,
                "triton": NS(cdiv=lambda n, d: (n + d - 1) // d),
            },
            filename=GRAPH_OPS,
        )
        self.namespace = {"torch": torch, "copy": copy.copy, "scatter_rows_": scatter}
        extract(source, "_get_full_attn_metadata", self.namespace)
        self.run = extract(source, "_draft_extend_topk", self.namespace, owner=OWNER)
        self.decode = Mock(return_value=torch.full((2, 11), 73, dtype=torch.int32))
        self.indexer = NS(
            _glm53_draft_graph_steps=4,
            index_topk=8,
            index_kpool=4,
            _decode_topk=self.decode,
        )
        # Distinct markers expose accidentally compacting request 2 behind the
        # accepted rows of request 1. Positions also cross the KPool ring wrap.
        self.q = torch.arange(8, dtype=torch.float32).view(8, 1)
        self.key = self.q + 100
        self.weights = self.q + 200
        self.gate = self.q + 300
        self.positions = torch.tensor([9, 10, 11, 12, 21, 22, 23, 24])
        self.tables = torch.tensor([[7, 8, 9, 10], [-1, -1, -1, -1]])

    def __call__(self, accepted, req=None):
        self.decode.reset_mock()
        self.launch.reset_mock()
        if req is None:
            req = torch.tensor([1, 2])
        self.batch = NS(
            batch_size=2,
            extend_seq_lens=torch.tensor([4, 4], dtype=torch.int32),
            extend_start_loc=torch.tensor([0, 4], dtype=torch.int32),
            req_pool_indices=req,
            spec_info=NS(num_accept_tokens=accepted),
            _ascend_kpool_metadata=NS(seq_lens=torch.tensor([13, 25])),
        )
        return self.run(
            self.indexer,
            self.q,
            self.key,
            self.weights,
            self.gate,
            self.positions,
            self.batch,
            self.tables,
            3,
        )

    def routing(self):
        return (
            torch.stack(
                [c.args[5].req_pool_indices for c in self.decode.call_args_list]
            ),
            torch.stack(
                [
                    c.args[5]._ascend_kpool_metadata.seq_lens
                    for c in self.decode.call_args_list
                ]
            ),
            torch.stack([c.args[2] for c in self.launch.call_args_list]),
        )


class DraftKPoolAcceptanceTests(unittest.TestCase):
    def assert_tensor(self, actual, expected):
        torch.testing.assert_close(
            actual, torch.as_tensor(expected, dtype=actual.dtype), rtol=0, atol=0
        )

    def test_partial_accept_masks_real_decode_and_scatter_calls(self):
        calls = Calls()
        output = calls(torch.tensor([1, 3], dtype=torch.int32))
        req, seq, dest = calls.routing()
        self.assert_tensor(req, [[1, 2], [0, 2], [0, 2], [0, 0]])
        self.assert_tensor((req > 0).sum(0), [1, 3])
        self.assert_tensor(seq, [[10, 22], [0, 23], [0, 24], [0, 0]])
        self.assert_tensor(dest, [[0, 4], [8, 5], [8, 6], [8, 8]])
        self.assertEqual(output.shape, (8, 11))
        self.assert_tensor(
            calls.batch._ascend_kpool_metadata.actual_seq_lengths_q, [4, 8]
        )
        self.assert_tensor(calls.batch.extend_start_loc, [0, 4])
        self.assert_tensor(calls.batch.req_pool_indices, [1, 2])
        for step, call in enumerate(calls.decode.call_args_list):
            rows = torch.tensor([step, 4 + step])
            for actual, source in zip(
                call.args[:5],
                (calls.q, calls.key, calls.weights, calls.gate, calls.positions),
            ):
                self.assert_tensor(actual, source.index_select(0, rows))
            self.assertIs(call.args[6], calls.tables)
            self.assertEqual(call.args[7], 3)

    def test_regression_is_observable_in_audited_baseline(self):
        old = Calls(BASE_SOURCE)
        old(torch.tensor([1, 3], dtype=torch.int32))
        req, _, dest = old.routing()
        self.assert_tensor(req, [[1, 2]] * 4)
        self.assert_tensor(dest, [[0, 4], [1, 5], [2, 6], [3, 7]])

    def test_full_accept_has_identical_baseline_launch_arguments(self):
        old, new = Calls(BASE_SOURCE), Calls()
        accepted = torch.tensor([4, 4], dtype=torch.int32)
        old(accepted)
        new(accepted)
        for actual, expected in zip(new.routing(), old.routing()):
            self.assert_tensor(actual, expected)
        for actual, expected in zip(
            new.decode.call_args_list, old.decode.call_args_list
        ):
            for a, b in zip(actual.args[:5], expected.args[:5]):
                self.assert_tensor(a, b)
        for actual, expected in zip(
            new.launch.call_args_list, old.launch.call_args_list
        ):
            for a, b in zip(actual.args, expected.args):
                self.assert_tensor(a, b)
            self.assertEqual(actual.kwargs, expected.kwargs)

    def test_graph_padding_with_full_accept_uses_only_sink(self):
        calls = Calls()
        calls(torch.tensor([3, 4], dtype=torch.int32), torch.tensor([1, 0]))
        req, seq, dest = calls.routing()
        self.assert_tensor(req, [[1, 0], [1, 0], [1, 0], [0, 0]])
        self.assert_tensor(seq[:, 1], [0, 0, 0, 0])
        self.assert_tensor(dest, [[0, 8], [1, 8], [2, 8], [8, 8]])

    def test_in_place_accept_changes_need_no_host_tensor_reads(self):
        calls = Calls()
        accepted = torch.tensor([4, 4], dtype=torch.int32)
        address = accepted.data_ptr()
        # Guard the production path, not the assertion/reporting helpers.
        with patch.object(
            torch.Tensor, "tolist", side_effect=AssertionError("tolist")
        ), patch.object(
            torch.Tensor, "item", side_effect=AssertionError("item")
        ), patch.object(
            torch.Tensor, "cpu", side_effect=AssertionError("cpu")
        ), patch.object(
            torch.Tensor, "__bool__", side_effect=AssertionError("host bool")
        ):
            calls(accepted)
            before = calls.routing()[0]
            accepted.copy_(torch.tensor([1, 3], dtype=torch.int32))
            calls(accepted)
            after = calls.routing()[0]
        self.assertEqual(accepted.data_ptr(), address)
        self.assert_tensor(before, [[1, 2]] * 4)
        self.assert_tensor(after, [[1, 2], [0, 2], [0, 2], [0, 0]])

    def test_cpu_trace_reads_changed_accept_and_padding_tensors(self):
        calls = Calls()

        def routing(accepted, req):
            calls(accepted, req)
            return calls.routing()

        # Trace the real function once with full acceptance. Mock launches
        # expose its output arguments as trace outputs; kernels do not run.
        traced = torch.jit.trace(
            routing,
            (torch.tensor([4, 4], dtype=torch.int32), torch.tensor([1, 2])),
            check_trace=False,
        )
        calls.decode.reset_mock()
        req, seq, dest = traced(
            torch.tensor([1, 3], dtype=torch.int32), torch.tensor([1, 2])
        )
        self.assert_tensor(req, [[1, 2], [0, 2], [0, 2], [0, 0]])
        self.assert_tensor(dest, [[0, 4], [8, 5], [8, 6], [8, 8]])
        req, seq, dest = traced(
            torch.tensor([3, 4], dtype=torch.int32), torch.tensor([1, 0])
        )
        self.assert_tensor(req, [[1, 0], [1, 0], [1, 0], [0, 0]])
        self.assert_tensor(seq[:, 1], [0, 0, 0, 0])
        self.assert_tensor(dest[:, 1], [8, 8, 8, 8])
        calls.decode.assert_not_called()  # Replay did not rerun Python routing.

    def test_non_draft_source_and_dispatch_are_unchanged(self):
        # Apart from the one draft method, the full production AST is identical
        # to the audited baseline (comments do not affect this comparison).
        def without_draft(source):
            tree = ast.parse(source)
            owner = next(
                n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == OWNER
            )
            owner.body = [
                n
                for n in owner.body
                if getattr(n, "name", None) != "_draft_extend_topk"
            ]
            return ast.dump(tree, include_attributes=False)

        self.assertEqual(without_draft(SOURCE), without_draft(BASE_SOURCE))
        for mode, expected in (
            (MODE.EXTEND, "_prefill_topk"),
            (MODE.DECODE, "_decode_topk"),
            (MODE.TARGET_VERIFY, "_verify_topk"),
            (MODE.IDLE, None),
        ):
            with self.subTest(mode=mode):
                calls = Calls()
                calls(torch.tensor([1, 3], dtype=torch.int32))
                calls.batch.forward_mode = mode
                # These modes have no num_accept_tokens contract.
                calls.batch.spec_info = None
                calls.batch._ascend_kpool_metadata.block_tables = calls.tables
                indexer = calls.indexer
                indexer.hidden_size = 1
                indexer.index_kpool_compress_gate = torch.ones((1, 1))
                indexer._project_q_key_weights = Mock(
                    return_value=(calls.q, calls.key, calls.weights)
                )
                routes = (
                    "_decode_topk",
                    "_draft_extend_topk",
                    "_verify_topk",
                    "_prefill_topk",
                )
                for name in routes:
                    setattr(
                        indexer,
                        name,
                        Mock(return_value=torch.ones((8, 11), dtype=torch.int32)),
                    )
                indexer._store_prefill_pools = Mock(return_value=[])
                namespace = dict(
                    calls.namespace,
                    F=torch.nn.functional,
                    __package__="_glm53_accept_test",
                    _as_ascend_sparse_indices=lambda x: x,
                )
                forward = extract(SOURCE, "forward_npu", namespace, owner=OWNER)
                compact = types.ModuleType("_glm53_accept_test.compact_index")
                compact.index_block_table = Mock(return_value=calls.tables)
                with patch.dict(sys.modules, {compact.__name__: compact}):
                    forward(indexer, calls.q, calls.q, calls.positions, calls.batch, 3)
                for name in routes:
                    self.assertEqual(
                        getattr(indexer, name).call_count, int(name == expected)
                    )


if __name__ == "__main__":
    unittest.main()
