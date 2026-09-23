"""Local partition/policy/oracle tests; no torch, NPU or distributed launch.

Run from the source repo:
    python test/manual/ascend/test_glm53_pd_prefill_index_cpu.py -v

The legacy policy regression reads the frozen b0e commit already in this repo.
Torch imports are stubbed only for scalar policy checks and verify_once's
comparison/logging contract. No CANN scorer or performance claim is made.
"""

import ast
import importlib.util
import itertools
import json
import os
import random
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from glm53_topology_test_schema import RUNTIME, SERVER_FIELDS, server_args_shape

ROOT = Path(__file__).resolve().parents[3]
SOURCE = (
    "python/sglang/srt/hardware_backend/npu/attention/glm53/prefill_index_parallel.py"
)
BASE = "b0e05ff2798a79ddcba4814324b1c03460694171"
SPEC = importlib.util.spec_from_file_location(
    "pd_prefill_index_under_test", ROOT / SOURCE
)
INDEX = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = INDEX
SPEC.loader.exec_module(INDEX)


class Mode:
    def __init__(self, name="extend"):
        self.name = name

    def is_extend_without_speculative(self):
        return self.name in ("extend", "mixed")

    def is_mixed(self):
        return self.name == "mixed"


def args(**changes):
    return server_args_shape(
        **(
            {
                "tp_size": 16,
                "ep_size": 16,
                "nnodes": 1,
                "pp_size": 1,
                "enable_dp_attention": False,
                "enable_two_batch_overlap": False,
                "disable_overlap_schedule": True,
                "quantization": "modelslim",
                "disable_radix_cache": True,
                "disaggregation_mode": "prefill",
                "dp_size": 1,
                "moe_dp_size": 1,
                "dwdp_size": 1,
                "attn_cp_size": 1,
                "dcp_size": 1,
                "enable_prefill_cp": False,
            }
            | changes
        )
    )


def batch(lengths=(4096,), ends=(65536,), mode="extend"):
    return types.SimpleNamespace(
        batch_size=len(lengths),
        extend_seq_lens_cpu=list(lengths),
        seq_lens_cpu=list(ends),
        forward_mode=Mode(mode),
    )


def query(device="npu", dtype="bf16", shape=(4096, 32, 128)):
    return types.SimpleNamespace(
        device=types.SimpleNamespace(type=device),
        dtype=dtype,
        ndim=len(shape),
        shape=shape,
    )


def oracle_inputs(forward_batch):
    tokens = sum(forward_batch.extend_seq_lens_cpu[: forward_batch.batch_size])
    q = types.SimpleNamespace(shape=(tokens, 32, 128))
    weights = types.SimpleNamespace(shape=(tokens, 32))
    compressed = [
        types.SimpleNamespace(shape=(int(end) // 4, 128))
        for end in forward_batch.seq_lens_cpu[: forward_batch.batch_size]
    ]
    return q, weights, compressed


class TensorValue:
    def __init__(self, values):
        self.values = tuple(values)
        self.shape = (len(values), 1)


class Owner:
    def __init__(self, layer_id, expected):
        self.layer_id, self.expected, self.calls = layer_id, expected, 0

    def _prefill_pooled_topk(self, *args):
        self.calls += 1
        return self.expected


def tiles_by_rows(request, offset, length, sequence_end, history_length):
    """Independent per-token grouping; captures GEMM shape and causal bounds."""
    first = sequence_end - length
    result = []
    for _, rows in itertools.groupby(range(length), key=lambda i: (first + i) // 128):
        rows = list(rows)
        result.append(
            (
                request,
                offset + rows[0],
                len(rows),
                first + rows[-1] + 1,
                min((first + rows[-1] + 1) // 4, history_length),
                (first + rows[0] + 1) // 4,
                (first + rows[-1] + 1) // 4,
            )
        )
    return result


class TestPartition(unittest.TestCase):
    def check_partition(self, lengths, ends, world):
        partitions, capacity = INDEX.partition_queries(lengths, ends, world)
        expected, offset = [], 0
        for req, (length, end) in enumerate(zip(lengths, ends)):
            expected.extend(tiles_by_rows(req, offset, length, end, end // 4))
            offset += length
        actual, rows = [], []
        self.assertEqual(len(partitions), world)
        for fragments in partitions:
            self.assertLessEqual(sum(f.length for f in fragments), capacity)
            for f in fragments:
                actual.extend(
                    tiles_by_rows(
                        f.request,
                        f.offset,
                        f.length,
                        f.sequence_end,
                        ends[f.request] // 4,
                    )
                )
                rows.extend(range(f.offset, f.offset + f.length))
        self.assertEqual(actual, expected)
        self.assertEqual(rows, list(range(sum(lengths))))

    def test_long_prefills_and_scheduler_chunks(self):
        for world in (1, 2, 4, 16):
            for lengths, ends in (
                ([65536], [65536]),
                ([131072], [131072]),
                ([4096], [65536]),
                ([4096], [131072]),
                ([4096, 4096], [65536, 131072]),
            ):
                with self.subTest(world=world, lengths=lengths, ends=ends):
                    self.check_partition(lengths, ends, world)

    def test_prefix_hits_at_and_around_causal_boundaries(self):
        for prefix in (
            0,
            1,
            3,
            4,
            63,
            64,
            127,
            128,
            129,
            255,
            256,
            257,
            65535,
            65536,
            65537,
            131071,
            131072,
        ):
            for length in (1, 3, 127, 128, 129, 2047, 2048, 4096):
                with self.subTest(prefix=prefix, length=length):
                    self.check_partition([length], [prefix + length], 16)

    def test_variable_requests_padding_and_empty_rank_partitions(self):
        for world in (1, 2, 4, 16):
            for lengths, ends in (
                ([257, 129, 0, 333], [65535, 129, 0, 131071]),
                ([127, 128, 129], [256, 256, 258]),
                ([3], [131075]),
                ([0, 0], [4, 128]),
                ([], []),
            ):
                self.check_partition(lengths, ends, world)

    def test_deterministic_random_partitions(self):
        rng = random.Random(20260915)
        for _ in range(100):
            lengths = [rng.randrange(0, 900) for _ in range(rng.randrange(1, 8))]
            ends = [length + rng.randrange(0, 131073) for length in lengths]
            self.check_partition(lengths, ends, rng.choice([1, 2, 4, 16]))

    def test_invalid_metadata(self):
        for lengths, ends, world in (
            ([-1], [0], 16),
            ([3], [2], 16),
            ([1], [], 16),
            ([1], [1], 0),
        ):
            with self.assertRaises(ValueError):
                INDEX.partition_queries(lengths, ends, world)


class TestFlagsAndOracle(unittest.TestCase):
    def setUp(self):
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.config = args()
        self.rank = 0
        fake_torch = types.ModuleType("torch")
        fake_torch.bfloat16 = "bf16"
        fake_torch.equal = lambda a, b: a.values == b.values
        fake_dist = types.ModuleType("torch.distributed")
        fake_dist.get_rank = lambda group: self.rank
        fake_dist.get_world_size = lambda group: 16
        fake_torch.distributed = fake_dist
        runtime = types.ModuleType("sglang.srt.runtime_context")
        runtime.get_server_args = lambda: self.config
        self.module_patch = patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "torch.distributed": fake_dist,
                "sglang.srt.runtime_context": runtime,
            },
        )
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def test_pd_flag_is_explicit_default_off_and_master_switch_still_wins(self):
        for value in (None, "0", "true", "2", ""):
            if value is None:
                os.environ.pop("SGLANG_GLM53_PD_PREFILL_INDEX_TP", None)
            else:
                os.environ["SGLANG_GLM53_PD_PREFILL_INDEX_TP"] = value
            self.assertFalse(INDEX.enabled(query(), batch()))
        os.environ["SGLANG_GLM53_PD_PREFILL_INDEX_TP"] = "1"
        self.assertTrue(INDEX.enabled(query(), batch()))
        os.environ["SGLANG_GLM53_PREFILL_INDEX_TP"] = "0"
        self.assertFalse(INDEX.enabled(query(), batch()))

    def test_prefix_on_requires_existing_verify_once_gate(self):
        os.environ["SGLANG_GLM53_PD_PREFILL_INDEX_TP"] = "1"
        self.config.disable_radix_cache = False
        for value in (None, ""):
            if value is None:
                os.environ.pop("SGLANG_GLM53_PREFILL_INDEX_VERIFY_DIR", None)
            else:
                os.environ["SGLANG_GLM53_PREFILL_INDEX_VERIFY_DIR"] = value
            self.assertFalse(INDEX.enabled(query(), batch()))
        os.environ["SGLANG_GLM53_PREFILL_INDEX_VERIFY_DIR"] = "/tmp/oracle"
        self.assertTrue(INDEX.enabled(query(), batch()))

    def test_decode_and_speculative_modes_stay_disabled(self):
        os.environ["SGLANG_GLM53_PD_PREFILL_INDEX_TP"] = "1"
        os.environ["SGLANG_GLM53_PREFILL_INDEX_VERIFY_DIR"] = "/tmp/oracle"
        for radix_disabled in (False, True):
            self.config = args(
                disaggregation_mode="decode", disable_radix_cache=radix_disabled
            )
            self.assertFalse(INDEX.enabled(query(), batch()))
        self.config = args()
        for mode in ("decode", "idle", "target_verify", "draft_extend_v2", "mixed"):
            self.assertFalse(INDEX.enabled(query(), batch(mode=mode)))

    def test_topology_schedule_quantization_guards(self):
        os.environ["SGLANG_GLM53_PD_PREFILL_INDEX_TP"] = "1"
        for key, value in {
            "tp_size": 8,
            "ep_size": 8,
            "nnodes": 2,
            "pp_size": 2,
            "enable_dp_attention": True,
            "enable_two_batch_overlap": True,
            "disable_overlap_schedule": False,
            "quantization": "bf16",
            "dp_size": 4,
            "moe_dp_size": 2,
            "dwdp_size": 16,
            "attn_cp_size": 2,
            "enable_prefill_cp": True,
        }.items():
            with self.subTest(key=key):
                self.config = args(**{key: value})
                self.assertFalse(INDEX.enabled(query(), batch()))

    def test_910b_tp8_ep8_prefill_index_is_opt_in_and_keeps_guards(self):
        self.config = args(tp_size=8, ep_size=8)
        self.assertFalse(INDEX.enabled(query(), batch()))
        os.environ["SGLANG_GLM53_PD_PREFILL_INDEX_TP"] = "1"
        self.assertTrue(INDEX.enabled(query(), batch()))
        self.config.disable_radix_cache = False
        self.assertFalse(INDEX.enabled(query(), batch()))
        os.environ["SGLANG_GLM53_PREFILL_INDEX_VERIFY_DIR"] = "/tmp/oracle"
        self.assertTrue(INDEX.enabled(query(), batch()))
        self.config.ep_size = 16
        self.assertFalse(INDEX.enabled(query(), batch()))
        self.config = args(tp_size=8, ep_size=8, disaggregation_mode="null")
        self.assertFalse(INDEX.enabled(query(), batch()))

    def test_only_audited_dp_topologies_are_admitted_for_verify(self):
        os.environ["SGLANG_GLM53_PD_PREFILL_INDEX_TP"] = "1"
        os.environ["SGLANG_GLM53_PREFILL_INDEX_VERIFY_DIR"] = "/tmp/oracle"
        for dp_size, attn_tp_size in ((2, 8), (4, 4)):
            with self.subTest(dp_size=dp_size, attn_tp_size=attn_tp_size):
                self.config = args(
                    dp_size=dp_size,
                    enable_dp_attention=True,
                    disable_radix_cache=False,
                )
                self.assertNotIn("attn_tp_size", SERVER_FIELDS)
                self.assertFalse(hasattr(self.config, "attn_tp_size"))
                self.assertEqual(RUNTIME["attn_tp_size_of"](self.config), attn_tp_size)
                self.assertTrue(INDEX.enabled(query(), batch()))
        for dp_size, attn_tp_size in ((2, 4), (4, 2)):
            with self.subTest(dp_size=dp_size):
                self.config = args(
                    tp_size=8,
                    ep_size=8,
                    dp_size=dp_size,
                    enable_dp_attention=True,
                    disable_radix_cache=False,
                )
                self.assertEqual(RUNTIME["attn_tp_size_of"](self.config), attn_tp_size)
                self.assertTrue(INDEX.enabled(query(), batch()))
        for mutation in (
            dict(dp_size=8, enable_dp_attention=True),
            dict(dp_size=0, enable_dp_attention=True),
            dict(dp_size=3, enable_dp_attention=True),
            dict(dp_size=8, tp_size=8, ep_size=8, enable_dp_attention=True),
            dict(dp_size=2, tp_size=8, ep_size=16, enable_dp_attention=True),
            dict(dp_size=2, tp_size=8, ep_size=8, nnodes=2, enable_dp_attention=True),
            dict(dp_size=4, tp_size=8, ep_size=16, enable_dp_attention=True),
            dict(dp_size=4, tp_size=8, ep_size=8, nnodes=2, enable_dp_attention=True),
            dict(dp_size=4, tp_size=8, ep_size=8, attn_cp_size=2, enable_dp_attention=True),
            dict(dp_size=4, tp_size=32, enable_dp_attention=True),
            dict(dp_size=2, enable_dp_attention=False),
            dict(dp_size=2, ep_size=8, enable_dp_attention=True),
            dict(dp_size=2, attn_cp_size=0, enable_dp_attention=True),
            dict(dp_size=2, enable_prefill_cp=True, enable_dp_attention=True),
            dict(
                dp_size=2,
                enable_dp_attention=True,
                moe_dp_size=2,
            ),
            dict(
                dp_size=2,
                enable_dp_attention=True,
                dwdp_size=2,
            ),
            dict(
                dp_size=2,
                enable_dp_attention=True,
                attn_cp_size=2,
            ),
            dict(
                dp_size=2,
                enable_dp_attention=True,
                pp_size=2,
            ),
            dict(
                dp_size=2,
                enable_dp_attention=True,
                nnodes=2,
            ),
        ):
            with self.subTest(mutation=mutation):
                self.config = args(disable_radix_cache=False, **mutation)
                self.assertFalse(INDEX.enabled(query(), batch()))

    def test_910b_dp2tp4_and_dp4tp2_keep_cache_and_performance_guards(self):
        os.environ["SGLANG_GLM53_PD_PREFILL_INDEX_TP"] = "1"
        for dp_size in (2, 4):
            with self.subTest(dp_size=dp_size):
                self.config = args(
                    tp_size=8, ep_size=8, dp_size=dp_size,
                    enable_dp_attention=True, disable_radix_cache=False,
                )
                os.environ.pop("SGLANG_GLM53_PREFILL_INDEX_VERIFY_DIR", None)
                os.environ.pop("SGLANG_GLM53_PD_PREFILL_INDEX_TP_MODE", None)
                os.environ.pop("SGLANG_GLM53_PD_PREFILL_INDEX_TP_DP_PERFORMANCE", None)
                self.assertFalse(INDEX.enabled(query(), batch()))
                os.environ["SGLANG_GLM53_PREFILL_INDEX_VERIFY_DIR"] = "/tmp/oracle"
                self.assertTrue(INDEX.enabled(query(), batch()))
                os.environ["SGLANG_GLM53_PD_PREFILL_INDEX_TP_MODE"] = "performance"
                self.assertFalse(INDEX.enabled(query(), batch()))
                os.environ["SGLANG_GLM53_PD_PREFILL_INDEX_TP_DP_PERFORMANCE"] = "1"
                self.assertTrue(INDEX.enabled(query(), batch()))
                self.assertFalse(INDEX.enabled(query(), batch(mode="mixed")))

    def test_server_args_shape_rejects_derived_width_injection(self):
        with self.assertRaises(AttributeError):
            args(attn_tp_size=8)
        with self.assertRaises(AttributeError):
            self.config.attn_tp_size = 8

    def test_dp_performance_requires_separate_exact_verified_unlock(self):
        os.environ["SGLANG_GLM53_PD_PREFILL_INDEX_TP"] = "1"
        os.environ["SGLANG_GLM53_PD_PREFILL_INDEX_TP_MODE"] = "performance"
        self.config = args(
            dp_size=2,
            enable_dp_attention=True,
            disable_radix_cache=False,
        )
        for value in (None, "", "0", "true", "2"):
            if value is None:
                os.environ.pop(
                    "SGLANG_GLM53_PD_PREFILL_INDEX_TP_DP_PERFORMANCE", None
                )
            else:
                os.environ[
                    "SGLANG_GLM53_PD_PREFILL_INDEX_TP_DP_PERFORMANCE"
                ] = value
            self.assertFalse(INDEX.enabled(query(), batch()))
        os.environ["SGLANG_GLM53_PD_PREFILL_INDEX_TP_DP_PERFORMANCE"] = "1"
        self.assertTrue(INDEX.enabled(query(), batch()))

    def test_tensor_and_work_threshold_guards(self):
        os.environ["SGLANG_GLM53_PD_PREFILL_INDEX_TP"] = "1"
        for q in (
            query(device="cpu"),
            query(dtype="fp32"),
            query(shape=(4096, 128)),
            query(shape=(4096, 16, 128)),
            query(shape=(4096, 32, 64)),
        ):
            self.assertFalse(INDEX.enabled(q, batch()))
        self.assertFalse(INDEX.enabled(query(), batch([2047], [65536])))
        self.assertTrue(INDEX.enabled(query(), batch([2048], [65536])))
        self.assertTrue(INDEX.enabled(query(), batch([1024, 1024], [65536, 131072])))

    def test_null_policy_matches_frozen_baseline(self):
        source = subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT}", "-C", str(ROOT),
             "show", f"{BASE}:{SOURCE}"], text=True
        )
        enabled_fn = next(
            n
            for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef) and n.name == "enabled"
        )
        namespace = {"os": os}
        exec(
            compile(
                ast.Module(body=[enabled_fn], type_ignores=[]),
                "baseline_enabled",
                "exec",
            ),
            namespace,
        )
        legacy = namespace["enabled"]
        mutations = (
            {},
            {"tp_size": 8},
            {"enable_dp_attention": True},
            {"disable_overlap_schedule": False},
            {"quantization": "bf16"},
            {"attn_cp_size": 2},
            {"dp_size": 4},
            {"enable_prefill_cp": True},
        )
        for master, pd_opt, radix, mutation in itertools.product(
            ("0", "1"), ("0", "1"), (False, True), mutations
        ):
            os.environ["SGLANG_GLM53_PREFILL_INDEX_TP"] = master
            os.environ["SGLANG_GLM53_PD_PREFILL_INDEX_TP"] = pd_opt
            self.config = args(
                disaggregation_mode="null", disable_radix_cache=radix, **mutation
            )
            for q, fb in (
                (query(), batch()),
                (query(device="cpu"), batch()),
                (query(), batch([2047], [65536])),
            ):
                self.assertEqual(INDEX.enabled(q, fb), legacy(q, fb))
        # The original null path must not start requiring new PD-only fields.
        self.config = args(disaggregation_mode="null")
        for key in (
            "attn_cp_size",
            "enable_prefill_cp",
            "dp_size",
            "moe_dp_size",
            "dwdp_size",
        ):
            delattr(self.config, key)
        os.environ["SGLANG_GLM53_PREFILL_INDEX_TP"] = "1"
        self.assertTrue(INDEX.enabled(query(), batch()))

    def test_oracle_checks_each_rank_layer_shape_and_records_absolute_ends(self):
        with tempfile.TemporaryDirectory() as directory:
            os.environ["SGLANG_GLM53_PREFILL_INDEX_VERIFY_DIR"] = directory
            output = TensorValue([11, 7, 3])
            for rank in range(16):
                self.rank = rank
                for layer in (0, 3, 11):
                    owner = Owner(layer, output)
                    for end in (65536, 131072):
                        fb = batch([4096], [end])
                        q, weights, compressed = oracle_inputs(fb)
                        INDEX.verify_once(
                            owner, output, q, weights, compressed, fb, None
                        )
                        INDEX.verify_once(
                            owner, output, q, weights, compressed, fb, None
                        )
                    self.assertEqual(owner.calls, 2)
                records = [
                    json.loads(line)
                    for line in (Path(directory) / f"rank{rank}.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertEqual(len(records), 6)
                self.assertTrue(all(r["exact"] and r["rank"] == rank for r in records))
                self.assertEqual(
                    {tuple(r["sequence_ends"]) for r in records}, {(65536,), (131072,)}
                )
                self.assertEqual({r["layer_id"] for r in records}, {0, 3, 11})

    def test_oracle_mismatch_raises_and_is_not_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            os.environ["SGLANG_GLM53_PREFILL_INDEX_VERIFY_DIR"] = directory
            owner = Owner(3, TensorValue([1, 2]))
            fb = batch()
            q, weights, compressed = oracle_inputs(fb)
            with self.assertRaisesRegex(RuntimeError, "disagrees"):
                INDEX.verify_once(
                    owner,
                    TensorValue([2, 1]),
                    q,
                    weights,
                    compressed,
                    fb,
                    None,
                )
            self.assertFalse(getattr(owner, "_glm53_checked_prefill_shapes", set()))
            record = json.loads((Path(directory) / "rank0.jsonl").read_text())
            self.assertFalse(record["exact"])
            INDEX.verify_once(
                owner, owner.expected, q, weights, compressed, fb, None
            )
            self.assertEqual(owner.calls, 2)

    def test_oracle_off_does_not_run_reference_scorer(self):
        owner = Owner(3, TensorValue([1]))
        INDEX.verify_once(owner, owner.expected, None, None, None, batch(), None)
        self.assertEqual(owner.calls, 0)


if __name__ == "__main__":
    unittest.main()
