"""Isolated tests for GLM-5.3 KPool tail-commit graph warmup.

The production module is loaded directly from its source file so collecting
these CPU tests does not import the rest of SGLang.  The default suite uses
real CPU ``torch.Tensor`` objects and mocks only the NPU graph boundary.

Run this file directly with ``--npu`` to add a small vendor-kernel/NPUGraph
check::

    python test_glm53_tail_commit_warmup.py --npu
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


_RUN_NPU_TESTS = "--npu" in sys.argv
if _RUN_NPU_TESTS:
    # Keep the explicit hardware switch out of unittest's argument parser.
    sys.argv.remove("--npu")

_REPO_ROOT = Path(__file__).resolve().parents[5]
_ACCEPTED_STATE_PATH = (
    _REPO_ROOT
    / "python/sglang/srt/hardware_backend/npu/attention/glm53/accepted_state.py"
)


def _load_accepted_state():
    spec = importlib.util.spec_from_file_location(
        "_glm53_accepted_state_under_test", _ACCEPTED_STATE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {_ACCEPTED_STATE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


accepted_state = _load_accepted_state()


def _make_indexer(scratch_capacity=8, request_slots=10, steps=4):
    return SimpleNamespace(
        _kpool_tail_k=torch.full(
            (request_slots, 2, 3), 11, dtype=torch.bfloat16
        ),
        _kpool_tail_score=torch.full(
            (request_slots, 2, 3), 13, dtype=torch.float32
        ),
        _kpool_mtp_tail_k=torch.full(
            (scratch_capacity, steps, 2, 3), 17, dtype=torch.bfloat16
        ),
        _kpool_mtp_tail_score=torch.full(
            (scratch_capacity, steps, 2, 3), 19, dtype=torch.float32
        ),
    )


def _make_model(indexers):
    return SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(self_attn=SimpleNamespace(indexer=indexer))
                for indexer in indexers
            ]
        )
    )


def _make_runner(
    *,
    local_capacity=6,
    capture_bs=(1, 4, 8),
    scratch_capacity=8,
    model=None,
    is_draft_worker=False,
    graph_enabled=True,
):
    if model is None:
        model = _make_model([_make_indexer(scratch_capacity=scratch_capacity)])
    return SimpleNamespace(
        is_draft_worker=is_draft_worker,
        decode_cuda_graph_runner=(
            SimpleNamespace(capture_bs=list(capture_bs)) if graph_enabled else None
        ),
        max_running_requests=local_capacity,
        model=model,
        attn_backend=SimpleNamespace(),
    )


def _recording_graph_type():
    class RecordingTailCommitGraph:
        instances = []

        def __init__(self, indexers, requests, accepted):
            self.indexers = tuple(indexers)
            self.initial_requests = requests.clone()
            self.initial_accepted = accepted.clone()
            self.replays = []
            type(self).instances.append(self)

        def replay(self, requests, accepted):
            self.replays.append((requests.clone(), accepted.clone()))

    return RecordingTailCommitGraph


class _FakeNpu:
    def __init__(self):
        self.synchronize = mock.Mock(name="npu_synchronize")
        self.captured_graph = SimpleNamespace(replay=mock.Mock(name="graph_replay"))

    def NPUGraph(self):
        return self.captured_graph

    def graph(self, graph):
        if graph is not self.captured_graph:
            raise AssertionError("unexpected graph passed to torch.npu.graph")
        return contextlib.nullcontext()


@contextlib.contextmanager
def _mock_scatter_module(scatter):
    root = types.ModuleType("sgl_kernel_npu")
    root.__path__ = []
    mamba = types.ModuleType("sgl_kernel_npu.mamba")
    mamba.__path__ = []
    leaf = types.ModuleType("sgl_kernel_npu.mamba.speculative_state_scatter")
    leaf.speculative_state_scatter_npu = scatter
    root.mamba = mamba
    mamba.speculative_state_scatter = leaf
    modules = {
        "sgl_kernel_npu": root,
        "sgl_kernel_npu.mamba": mamba,
        "sgl_kernel_npu.mamba.speculative_state_scatter": leaf,
    }
    with mock.patch.dict(sys.modules, modules):
        yield


class TestGlm53TailCommitWarmup(unittest.TestCase):
    def test_covers_every_raw_size_including_graph_bucket_holes(self):
        runner = _make_runner(local_capacity=6, capture_bs=(1, 4, 8))
        graph_type = _recording_graph_type()
        fake_npu = _FakeNpu()

        with (
            mock.patch.dict(os.environ, {"SGLANG_GLM53_MTP_COMMIT_GRAPH": "1"}),
            mock.patch.object(accepted_state, "_TailCommitGraph", graph_type),
            mock.patch.object(torch, "npu", fake_npu, create=True),
        ):
            summary = accepted_state.prewarm_kpool_tail_commit_graphs(runner)

        expected_sizes = list(range(1, min(6, 8) + 1))
        self.assertEqual(summary["batch_sizes"], expected_sizes)
        self.assertEqual(
            list(runner.attn_backend._glm53_tail_commit_graphs), expected_sizes
        )
        self.assertTrue(runner.attn_backend._glm53_tail_commit_prewarmed)
        self.assertEqual(len(graph_type.instances), len(expected_sizes))
        # 2, 3, 5 and 6 are absent from capture_bs but are valid raw batches.
        self.assertEqual(
            set(expected_sizes) - set(runner.decode_cuda_graph_runner.capture_bs),
            {2, 3, 5, 6},
        )
        for batch_size, graph in zip(expected_sizes, graph_type.instances):
            self.assertEqual(tuple(graph.initial_requests.shape), (batch_size,))
            self.assertEqual(graph.initial_requests.dtype, torch.int32)
            self.assertTrue(
                torch.equal(
                    graph.initial_requests, -torch.ones_like(graph.initial_requests)
                )
            )
            self.assertTrue(
                torch.equal(
                    graph.initial_accepted, -torch.ones_like(graph.initial_accepted)
                )
            )
        fake_npu.synchronize.assert_called_once_with()

    def test_prewarm_skips_unsupported_states_without_publishing(self):
        cases = (
            ("no_model", _make_runner(model=SimpleNamespace())),
            (
                "layer_without_self_attn",
                _make_runner(
                    model=SimpleNamespace(
                        model=SimpleNamespace(layers=[SimpleNamespace()])
                    )
                ),
            ),
            ("graph_disabled", _make_runner(graph_enabled=False)),
            ("draft", _make_runner(is_draft_worker=True)),
        )
        for name, runner in cases:
            with self.subTest(name=name):
                graph_type = _recording_graph_type()
                fake_npu = _FakeNpu()
                with (
                    mock.patch.dict(
                        os.environ, {"SGLANG_GLM53_MTP_COMMIT_GRAPH": "1"}
                    ),
                    mock.patch.object(
                        accepted_state, "_TailCommitGraph", graph_type
                    ),
                    mock.patch.object(torch, "npu", fake_npu, create=True),
                ):
                    result = accepted_state.prewarm_kpool_tail_commit_graphs(runner)
                self.assertIsNone(result)
                self.assertFalse(
                    hasattr(runner.attn_backend, "_glm53_tail_commit_graphs")
                )
                self.assertFalse(
                    hasattr(runner.attn_backend, "_glm53_tail_commit_prewarmed")
                )
                self.assertEqual(graph_type.instances, [])
                fake_npu.synchronize.assert_not_called()

        # The flag check must short-circuit before any runner/NPU attribute read.
        fake_npu = _FakeNpu()
        with (
            mock.patch.dict(os.environ, {"SGLANG_GLM53_MTP_COMMIT_GRAPH": "0"}),
            mock.patch.object(torch, "npu", fake_npu, create=True),
        ):
            self.assertIsNone(
                accepted_state.prewarm_kpool_tail_commit_graphs(SimpleNamespace())
            )
        fake_npu.synchronize.assert_not_called()

    def test_prewarm_is_idempotent(self):
        runner = _make_runner(local_capacity=4, capture_bs=(1, 4, 8))
        graph_type = _recording_graph_type()
        fake_npu = _FakeNpu()
        with (
            mock.patch.dict(os.environ, {"SGLANG_GLM53_MTP_COMMIT_GRAPH": "1"}),
            mock.patch.object(accepted_state, "_TailCommitGraph", graph_type),
            mock.patch.object(torch, "npu", fake_npu, create=True),
        ):
            first = accepted_state.prewarm_kpool_tail_commit_graphs(runner)
            registry = runner.attn_backend._glm53_tail_commit_graphs
            second = accepted_state.prewarm_kpool_tail_commit_graphs(runner)

        self.assertEqual(first["batch_sizes"], [1, 2, 3, 4])
        self.assertIsNone(second)
        self.assertIs(runner.attn_backend._glm53_tail_commit_graphs, registry)
        self.assertEqual(len(graph_type.instances), 4)
        fake_npu.synchronize.assert_called_once_with()

    def test_warmed_commit_reuses_graph_and_forwards_runtime_values(self):
        runner = _make_runner(local_capacity=4, capture_bs=(1, 4, 8))
        graph_type = _recording_graph_type()
        fake_npu = _FakeNpu()
        with (
            mock.patch.dict(os.environ, {"SGLANG_GLM53_MTP_COMMIT_GRAPH": "1"}),
            mock.patch.object(accepted_state, "_TailCommitGraph", graph_type),
            mock.patch.object(torch, "npu", fake_npu, create=True),
        ):
            accepted_state.prewarm_kpool_tail_commit_graphs(runner)
            construction_count = len(graph_type.instances)
            requests = torch.tensor([5, 2, 4], dtype=torch.int32)
            accepted = torch.tensor([3, 0, 1], dtype=torch.int32)
            requests_before = requests.clone()
            accepted_before = accepted.clone()
            with mock.patch.object(
                accepted_state,
                "_copy_tails",
                side_effect=AssertionError("warmed shape must not use eager fallback"),
            ):
                accepted_state.commit_kpool_tails(
                    runner.attn_backend, runner.model, accepted, requests
                )

        self.assertEqual(len(graph_type.instances), construction_count)
        warmed_graph = runner.attn_backend._glm53_tail_commit_graphs[3]
        self.assertEqual(len(warmed_graph.replays), 1)
        replay_requests, replay_accepted = warmed_graph.replays[0]
        self.assertTrue(torch.equal(replay_requests, requests_before))
        self.assertTrue(torch.equal(replay_accepted, accepted_before))
        self.assertTrue(torch.equal(requests, requests_before))
        self.assertTrue(torch.equal(accepted, accepted_before))

    def test_missing_shape_uses_eager_fallback_without_graph_construction(self):
        indexer = _make_indexer()
        model = _make_model([indexer])
        backend = SimpleNamespace(_glm53_tail_commit_graphs={1: object()})
        requests = torch.tensor([6, 3, 2], dtype=torch.int64)
        accepted = torch.tensor([2, 1, 0], dtype=torch.int64)
        sentinel = object()

        with (
            mock.patch.dict(os.environ, {"SGLANG_GLM53_MTP_COMMIT_GRAPH": "1"}),
            mock.patch.object(
                accepted_state, "_TailCommitGraph"
            ) as graph_constructor,
            mock.patch.object(
                accepted_state, "_copy_tails", return_value=sentinel
            ) as eager_copy,
        ):
            result = accepted_state.commit_kpool_tails(
                backend, model, accepted, requests
            )

        self.assertIs(result, sentinel)
        graph_constructor.assert_not_called()
        eager_copy.assert_called_once()
        call_indexers, call_requests, call_accepted = eager_copy.call_args.args
        self.assertEqual(call_indexers, [indexer])
        self.assertEqual(call_requests.dtype, torch.int32)
        self.assertEqual(call_accepted.dtype, torch.int32)
        self.assertTrue(torch.equal(call_requests, requests.to(torch.int32)))
        self.assertTrue(torch.equal(call_accepted, accepted.to(torch.int32)))
        self.assertTrue(torch.equal(requests, torch.tensor([6, 3, 2])))
        self.assertTrue(torch.equal(accepted, torch.tensor([2, 1, 0])))

    def test_invalid_capacity_fails_before_registry_is_published(self):
        cases = (
            ("non_positive", _make_runner(local_capacity=0, scratch_capacity=8)),
            ("scratch_too_small", _make_runner(local_capacity=4, scratch_capacity=3)),
        )
        for name, runner in cases:
            with self.subTest(name=name):
                graph_type = _recording_graph_type()
                fake_npu = _FakeNpu()
                with (
                    mock.patch.dict(
                        os.environ, {"SGLANG_GLM53_MTP_COMMIT_GRAPH": "1"}
                    ),
                    mock.patch.object(
                        accepted_state, "_TailCommitGraph", graph_type
                    ),
                    mock.patch.object(torch, "npu", fake_npu, create=True),
                ):
                    with self.assertRaises(ValueError):
                        accepted_state.prewarm_kpool_tail_commit_graphs(runner)

                self.assertFalse(
                    hasattr(runner.attn_backend, "_glm53_tail_commit_graphs")
                )
                self.assertFalse(
                    hasattr(runner.attn_backend, "_glm53_tail_commit_prewarmed")
                )
                self.assertEqual(graph_type.instances, [])
                fake_npu.synchronize.assert_not_called()

    def test_negative_warmup_indices_obey_no_write_scatter_contract(self):
        indexer = _make_indexer(scratch_capacity=3)
        before_k = indexer._kpool_tail_k.clone()
        before_score = indexer._kpool_tail_score.clone()
        calls = []

        def scatter(destination, source, requests, source_rows, accepted):
            calls.append(
                (
                    requests.clone(),
                    source_rows.clone(),
                    accepted.clone(),
                )
            )
            # CPU stand-in for the vendor contract.  Negative destination or
            # step indices must not read source or write destination.
            for destination_row, source_row, step in zip(
                requests.tolist(), source_rows.tolist(), accepted.tolist()
            ):
                if destination_row >= 0 and source_row >= 0 and step >= 0:
                    destination[0, destination_row].copy_(
                        source[0, source_row, step]
                    )

        fake_npu = _FakeNpu()
        requests = torch.full((3,), -1, dtype=torch.int32)
        accepted = torch.full_like(requests, -1)
        with (
            _mock_scatter_module(scatter),
            mock.patch.object(torch, "npu", fake_npu, create=True),
        ):
            graph = accepted_state._TailCommitGraph([indexer], requests, accepted)

        # _TailCommitGraph executes one eager warmup and one capture body;
        # each invokes the scatter once for K and once for score.
        self.assertEqual(len(calls), 4)
        for call_requests, source_rows, call_accepted in calls:
            self.assertTrue(torch.equal(call_requests, requests))
            self.assertTrue(
                torch.equal(source_rows, torch.arange(3, dtype=torch.int32))
            )
            self.assertTrue(torch.equal(call_accepted, accepted))
        self.assertTrue(torch.equal(indexer._kpool_tail_k, before_k))
        self.assertTrue(torch.equal(indexer._kpool_tail_score, before_score))
        self.assertTrue(torch.equal(graph.requests, requests))
        self.assertTrue(torch.equal(graph.accepted, accepted))
        fake_npu.synchronize.assert_called_once_with()


@unittest.skipUnless(_RUN_NPU_TESTS, "run this file directly with --npu")
class TestGlm53TailCommitWarmupNpu(unittest.TestCase):
    def test_vendor_scatter_graph_masks_negative_and_copies_positive_steps(self):
        # Importing torch_npu is intentionally confined to explicit hardware
        # mode so default CPU collection never initializes the NPU runtime.
        import torch_npu  # noqa: F401

        torch.npu.set_device(0)
        torch.manual_seed(20260920)
        indexer = SimpleNamespace()
        for suffix, dtype in (("k", torch.bfloat16), ("score", torch.float32)):
            setattr(
                indexer,
                "_kpool_tail_" + suffix,
                torch.randn(6, 4, 128, device="npu", dtype=dtype),
            )
            setattr(
                indexer,
                "_kpool_mtp_tail_" + suffix,
                torch.randn(2, 4, 4, 128, device="npu", dtype=dtype),
            )

        negative_requests = torch.full((2,), -1, device="npu", dtype=torch.int32)
        negative_accepted = torch.full_like(negative_requests, -1)
        before = {
            suffix: getattr(indexer, "_kpool_tail_" + suffix).clone()
            for suffix in ("k", "score")
        }
        graph = accepted_state._TailCommitGraph(
            [indexer], negative_requests, negative_accepted
        )
        torch.npu.synchronize()
        for suffix in ("k", "score"):
            torch.testing.assert_close(
                getattr(indexer, "_kpool_tail_" + suffix),
                before[suffix],
                rtol=0.0,
                atol=0.0,
            )

        requests = torch.tensor([3, 1], device="npu", dtype=torch.int32)
        accepted = torch.tensor([0, 2], device="npu", dtype=torch.int32)
        graph.replay(requests, accepted)
        torch.npu.synchronize()
        for suffix in ("k", "score"):
            expected = before[suffix].clone()
            source = getattr(indexer, "_kpool_mtp_tail_" + suffix)
            expected[3].copy_(source[0, 0])
            expected[1].copy_(source[1, 2])
            torch.testing.assert_close(
                getattr(indexer, "_kpool_tail_" + suffix),
                expected,
                rtol=0.0,
                atol=0.0,
            )


if __name__ == "__main__":
    unittest.main()
