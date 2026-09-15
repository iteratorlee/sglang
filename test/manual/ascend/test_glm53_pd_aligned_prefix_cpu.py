"""Local CPU/source checks for opt-in GLM NPU PD final-page checkpoints.

No SGLang runtime import, torch/NPU, network, or model launch. Extract real
scheduler/metadata/cache methods with AST and run them with NumPy tensor stand-ins;
a small independent recurrence checks boundary/resume bookkeeping, not NPU math.

    python -B test/manual/ascend/test_glm53_pd_aligned_prefix_cpu.py -v
"""

import ast
import copy
import enum
import os
import subprocess
import types
import unittest
from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = "ba07ed2a03dd1373161477cb2a38c1181fc96cd3"
SCHEDULE = "python/sglang/srt/managers/schedule_batch.py"
HYBRID = "python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py"
KDA = "python/sglang/srt/hardware_backend/npu/attention/ascend_kda_backend.py"
FB = "python/sglang/srt/model_executor/forward_batch_info.py"
LEGACY = "python/sglang/srt/mem_cache/mamba_radix_cache.py"
UNIFIED = "python/sglang/srt/mem_cache/unified_radix_cache.py"
COMPONENT = "python/sglang/srt/mem_cache/unified_cache/components/mamba.py"
POOL = "python/sglang/srt/mem_cache/memory_pool.py"
FLAG = "SGLANG_GLM53_PD_PREFILL_REUSABLE_CHECKPOINT"
NS = types.SimpleNamespace


def text(path, baseline=False):
    if baseline:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "show", f"{BASE}:{path}"], text=True
        )
    return (ROOT / path).read_text()


def node(path, name, baseline=False):
    matches = [
        n
        for n in ast.walk(ast.parse(text(path, baseline)))
        if isinstance(n, ast.FunctionDef) and n.name == name
    ]
    if len(matches) != 1:
        raise AssertionError((path, name, len(matches)))
    return matches[0]


def extract(path, name, namespace, baseline=False):
    definition = copy.deepcopy(node(path, name, baseline))
    definition.decorator_list = []
    mod = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            definition,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(mod), str(ROOT / path), "exec"), namespace)
    return namespace[name]


class Tensor(np.ndarray):
    def __new__(cls, value, dtype=None):
        return np.asarray(value, dtype=dtype).view(cls)

    def __getitem__(self, index):
        return Tensor(super().__getitem__(index))

    def cpu(self):
        return self

    def to(self, dtype=None, **kwargs):
        if isinstance(dtype, str):
            dtype = None
        return Tensor(self, dtype).copy() if kwargs.get("copy") else Tensor(self, dtype)

    def unsqueeze(self, axis):
        return Tensor(np.expand_dims(self, axis))

    def clone(self):
        return Tensor(self.copy())

    def clamp(self, low, high):
        return Tensor(np.clip(self, low, high))

    def nonzero(self, as_tuple=False):
        result = tuple(Tensor(x) for x in np.asarray(self).nonzero())
        return result if as_tuple else Tensor(np.column_stack(result))


TORCH = NS(
    arange=lambda *a, **kw: Tensor(np.arange(*a, dtype=kw.get("dtype"))),
    zeros_like=lambda x: Tensor(np.zeros_like(x)),
    cumsum=lambda x, dim: Tensor(np.cumsum(x, axis=dim)),
    full=lambda shape, value, **kw: Tensor(
        np.full(shape, value, dtype=kw.get("dtype"))
    ),
    int32=np.int32,
    int64=np.int64,
)


class Mode(enum.Enum):
    EXTEND = 1
    DECODE = 2
    TARGET_VERIFY = 3
    DRAFT_EXTEND_V2 = 4
    MIXED = 5
    SPLIT_PREFILL = 6


Entry = namedtuple("Entry", "track_mask track_index track_seqlen")


def environment(**overrides):
    cfg = dict(role="prefill", enabled=True, lazy=False, dp=1, pp=1, cp=1, dcp=1)
    cfg.update(overrides)
    return dict(
        os=os,
        torch=TORCH,
        ForwardMode=Mode,
        get_disagg=lambda: NS(disaggregation_mode=cfg["role"]),
        get_exec=lambda: NS(
            mamba=NS(
                enable_mamba_extra_buffer=cfg["enabled"],
                enable_mamba_extra_buffer_lazy=cfg["lazy"],
            )
        ),
        get_parallel=lambda: NS(
            dp_size=cfg["dp"],
            pp_size=cfg["pp"],
            attn_cp_size=cfg["cp"],
            dcp_size=cfg["dcp"],
        ),
        mamba_cache_chunk_size=lambda: 64,
        mamba_checkpoint_grid=lambda p: int(np.lcm(64, p)),
        _MambaRadixCacheV2TrackEntry=Entry,
    )


def make_request(end=4096, start=0, origin=None, output=(), branch=None, logprob=-1):
    req = NS(
        origin_input_ids=list(range(end if origin is None else origin)),
        output_ids=list(output),
        prefix_indices=Tensor(np.arange(start), np.int64),
        extend_range=NS(start=start, end=end, length=end - start),
        mamba_branching_seqlen=branch,
        return_logprob=logprob >= 0,
        logprob_start_len=logprob,
        kv=NS(
            mamba_ping_pong_track_buffer=Tensor([10, 11], np.int64),
            mamba_next_track_idx=0,
            mamba_last_track_idx=None,
            mamba_last_track_seqlen=None,
            mamba_pool_idx=Tensor(3),
            req_pool_idx=0,
            cache_protected_len=start,
        ),
        extra_key=None,
        cache_salt=None,
        kv_rotation_base=0,
        priority=0,
    )
    req._compute_max_prefix_len = types.MethodType(
        extract(SCHEDULE, "_compute_max_prefix_len", {}), req
    )
    req.get_fill_ids = lambda: req.origin_input_ids[:end]
    return req


def make_batch(
    device="npu:0",
    arch="Glm5NextForConditionalGeneration",
    mode=Mode.EXTEND,
    disabled=False,
    page=64,
    state_chunk=64,
):
    return NS(
        device=device,
        forward_mode=mode,
        model_config=NS(
            hf_config=NS(architectures=[arch]),
            hf_text_config=NS(mamba_chunk_size=state_chunk),
        ),
        tree_cache=NS(page_size=page, disable=disabled),
        req_to_token_pool=NS(get_mamba_ping_pong_other_idx=lambda i: 1 - i),
    )


def prepare(req, batch=None, env=None, enabled=True, baseline=False):
    env = environment() if env is None else env
    method = extract(
        SCHEDULE, "_mamba_radix_cache_v2_req_prepare_for_extend", env, baseline
    )
    with patch.dict(os.environ, {FLAG: "1" if enabled else "0"}):
        return method(make_batch() if batch is None else batch, req)


def metadata(requests, entries, window):
    env = dict(
        torch=TORCH,
        mamba_cache_chunk_size=lambda: 64,
        Mamba2AttnBackend=type("OtherBackend", (), {}),
    )
    fb = NS(
        mamba_track_mask=Tensor([e.track_mask for e in entries], bool),
        mamba_track_seqlens=Tensor([e.track_seqlen for e in entries], np.int64),
        mamba_track_indices=Tensor([e.track_index for e in entries], np.int64),
        extend_prefix_lens=Tensor([len(r.prefix_indices) for r in requests], np.int64),
        extend_seq_lens=Tensor([r.extend_range.length for r in requests], np.int64),
    )
    fb.mamba_track_aligned_lens = types.MethodType(
        extract(FB, "mamba_track_aligned_lens", env), fb
    )
    backend = NS(conv_states_shape=(32, 6, window), device="cpu", mamba_chunk_size=64)
    starts = Tensor([0] + list(np.cumsum(fb.extend_seq_lens)), np.int64)
    conv = extract(HYBRID, "_init_track_conv_indices", env)(backend, starts, fb)
    ssm = extract(HYBRID, "_init_track_ssm_indices", env)(
        backend, Tensor([3] * len(entries)), fb
    )
    return fb.mamba_track_aligned_lens(), conv, ssm


class InsertSeen(Exception):
    def __init__(self, params):
        self.params = params


class Key:
    def __init__(self, tokens, *args, **kwargs):
        self.tokens = list(tokens)

    def __len__(self):
        return len(self.tokens)

    def page_aligned(self, size):
        return Key(self.tokens[: len(self.tokens) // size * size])


def inserted(req, unified):
    """Execute production donation + real pre-insert cache path, intercept insert."""
    pool = NS(
        req_to_token=Tensor([req.get_fill_ids()], np.int64),
        mamba_pool=NS(replayssm_spec_write_pos=None),
        get_mamba_ping_pong_keep_idx=lambda r: r.kv.mamba_last_track_idx,
    )
    pool.set_mamba_ping_pong_slot = (
        lambda r, idx, v: r.kv.mamba_ping_pong_track_buffer.__setitem__(idx, v)
    )
    pool.donate_mamba_ping_pong_slot = types.MethodType(
        extract(POOL, "donate_mamba_ping_pong_slot", dict(_MAMBA_DEBUG_ASSERTS=False)),
        pool,
    )
    cache = NS(
        disable=False,
        enable_mamba_extra_buffer=True,
        req_to_token_pool=pool,
        page_size=64,
        int8_ckpt_pool=None,
        _alloc_mamba_slot=lambda: Tensor([20]),
        tree_core=NS(is_eagle=False),
        session=NS(try_cache_unfinished_req=lambda *a, **kw: False),
    )

    def catch(params):
        raise InsertSeen(params)

    cache.insert = catch
    env = dict(
        torch=TORCH,
        InsertParams=NS,
        RadixKey=Key,
        envs=NS(
            SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS=NS(get=lambda: False)
        ),
    )
    if unified:
        component = NS(
            cache=cache, int8_ckpt_pool=None, _alloc_mamba_slot=cache._alloc_mamba_slot
        )
        component.prepare_for_caching_req = types.MethodType(
            extract(COMPONENT, "prepare_for_caching_req", env), component
        )
        cache._components_tuple = (component,)
    method = extract(UNIFIED if unified else LEGACY, "cache_unfinished_req", env)
    try:
        method(cache, req)
    except InsertSeen as seen:
        return seen.params
    raise AssertionError("Cache path did not reach insert")


class AlignedPrefixTests(unittest.TestCase):
    def test_aligned_final_boundaries(self):
        for end, start in [(128, 0), (4096, 0), (65536, 61440), (131072, 126976)]:
            with self.subTest(end=end):
                req = make_request(end, start)
                entry = prepare(req)
                self.assertEqual(entry.track_seqlen, end - 64 + 1)
                self.assertEqual(req.kv.mamba_last_track_seqlen, end - 64)
                self.assertEqual(entry.track_index, 10)
                self.assertEqual(req.kv.mamba_last_track_idx, 0)
                self.assertEqual(req.kv.mamba_next_track_idx, 1)
                self.assertEqual(req.kv.mamba_pool_idx.item(), 3)

    def test_disabled_matches_base(self):
        for end in (1, 63, 64, 65, 127, 128, 255, 256, 4095, 4096, 4097, 65536, 131072):
            a, b = make_request(end), make_request(end)
            self.assertEqual(prepare(a, enabled=False), prepare(b, baseline=True))
            self.assertEqual(a.kv.mamba_last_track_seqlen, b.kv.mamba_last_track_seqlen)

    def test_opt_in_requires_literal_one(self):
        method = extract(
            SCHEDULE, "_mamba_radix_cache_v2_req_prepare_for_extend", environment()
        )
        for value in (None, "", "0", "true", "True", "2"):
            with self.subTest(value=value), patch.dict(os.environ):
                if value is None:
                    os.environ.pop(FLAG, None)
                else:
                    os.environ[FLAG] = value
                req, reference = make_request(), make_request()
                actual = method(make_batch(), req)
                expected = prepare(reference, baseline=True)
                self.assertEqual(actual, expected)
                self.assertEqual(
                    req.kv.mamba_last_track_seqlen, reference.kv.mamba_last_track_seqlen
                )

    def test_noneligible_matches_base(self):
        cases = [
            dict(batch=make_batch(device="cuda")),
            dict(batch=make_batch(device="cpu")),
            dict(batch=make_batch(arch="KimiLinearForCausalLM")),
            dict(batch=make_batch(disabled=True)),
            dict(batch=make_batch(page=128)),
            dict(batch=make_batch(state_chunk=128)),
            *[dict(batch=make_batch(mode=m)) for m in Mode if m != Mode.EXTEND],
            *[dict(env=environment(role=r)) for r in ("null", "decode")],
            *[
                dict(env=environment(**{k: v}))
                for k, v in (
                    ("lazy", True),
                    ("enabled", False),
                    ("dp", 2),
                    ("pp", 2),
                    ("cp", 2),
                    ("dcp", 2),
                )
            ],
        ]
        for options in cases:
            a, b = make_request(), make_request()
            self.assertEqual(
                prepare(a, **options), prepare(b, baseline=True, **options)
            )
            self.assertEqual(a.kv.mamba_last_track_seqlen, b.kv.mamba_last_track_seqlen)

    def test_intermediate_short_non_aligned_and_output_unchanged(self):
        for kw in [
            dict(end=4096, origin=65536),
            dict(end=4095),
            dict(end=4097),
            dict(end=64),
            dict(end=65536, start=65472),
            dict(end=4096, output=(1,)),
            dict(end=4096, logprob=1024),
            dict(end=4096, start=3),
        ]:
            with self.subTest(kw=kw):
                a, b = make_request(**kw), make_request(**kw)
                self.assertEqual(prepare(a), prepare(b, baseline=True))
                self.assertEqual(
                    a.kv.mamba_last_track_seqlen, b.kv.mamba_last_track_seqlen
                )

    def test_existing_branch_checkpoint_priority(self):
        for branch in (64, 1024, 4032):
            req = make_request(branch=branch)
            entry = prepare(req)
            self.assertEqual(req.kv.mamba_last_track_seqlen, branch)
            self.assertEqual(entry.track_seqlen, branch + 1)

    def test_real_metadata_conv_and_ssm_share_boundary(self):
        requests = [
            make_request(4096),
            make_request(65536, 61440),
            make_request(131072, 126976),
            make_request(4095),
        ]
        entries = [prepare(r) for r in requests]
        for window in (3, 6):
            lens, conv, ssm = metadata(requests, entries, window)
            np.testing.assert_array_equal(lens, [4032] * 4)
            for i, r in enumerate(requests):
                flattened_start = sum(q.extend_range.length for q in requests[:i])
                np.testing.assert_array_equal(
                    conv[i],
                    np.arange(flattened_start + 4032 - window, flattened_start + 4032),
                )
                self.assertEqual(
                    len(r.prefix_indices) + int(lens[i]), r.kv.mamba_last_track_seqlen
                )
            self.assertEqual(
                len(ssm[4]), 0
            )  # no final-state copy for end-64+1 sentinel
            np.testing.assert_array_equal(
                ssm[0], [63] * 4
            )  # real intermediate h selector

    def test_real_tree_insertion_and_donation(self):
        for unified in (False, True):
            for end, start in ((4096, 0), (65536, 61440), (131072, 126976)):
                with self.subTest(unified=unified, end=end):
                    req = make_request(end, start)
                    entry = prepare(req)
                    params = inserted(req, unified)
                    self.assertEqual(len(params.key), end - 64)
                    self.assertEqual(len(params.value), end - 64)
                    self.assertEqual(params.mamba_value.item(), entry.track_index)
                    self.assertEqual(
                        req.kv.mamba_pool_idx.item(), 3
                    )  # active transfer slot untouched
                    self.assertEqual(req.kv.mamba_ping_pong_track_buffer[0].item(), 20)

    def test_last_token_limit_reproduces_missing_checkpoint(self):
        req = make_request()
        limit = req._compute_max_prefix_len(4096)
        self.assertEqual(limit, 4095)

        def reusable(depths):
            return max([d for d in depths if d <= limit // 64 * 64] + [0])

        self.assertEqual(reusable([4096]), 0)
        self.assertEqual(reusable([4032]), 4032)
        req2 = make_request()
        prepare(req2)
        self.assertEqual(reusable([req2.kv.mamba_last_track_seqlen]), 4032)

    def test_cpu_recurrence_and_conv_resume_preserve_full_live_state(self):
        # Independent tiny FP32 delta recurrence after a causal convolution.
        # Actual NPU kernels are NOT executed by this test.
        for end, start in [(128, 0), (4096, 0), (65536, 61440), (131072, 126976)]:
            req = make_request(end, start)
            entry = prepare(req)
            lens, conv, _ = metadata([req], [entry], 6)
            cut = start + int(lens[0])
            raw = (
                (np.arange(end * 6, dtype=np.float32).reshape(end, 6) % 97) - 48
            ) / 64

            def convolve(rows, history):
                joined = np.concatenate([history[-2:], rows])
                return (
                    joined[:-2] * np.float32(0.1)
                    + joined[1:-1] * np.float32(0.3)
                    + joined[2:] * np.float32(0.6)
                )

            def step(state, x):
                k = x[:2]
                k = k / np.sqrt(np.dot(k, k) + np.float32(1e-6))
                v = x[2:4]
                state = state * np.float32(0.97)
                return state + np.outer(k, (v - k @ state) * np.float32(0.4))

            values = convolve(raw, np.zeros((2, 6), np.float32))
            state = np.zeros((2, 2), np.float32)
            snapshot = None
            for i, x in enumerate(values):
                state = step(state, x)
                if i + 1 == cut:
                    snapshot = state.copy()
            full = state.copy()
            tracked_conv = raw[start + np.asarray(conv[0])].copy()
            np.testing.assert_array_equal(tracked_conv, raw[cut - 6 : cut])
            resumed_values = convolve(raw[cut:], tracked_conv)
            np.testing.assert_array_equal(resumed_values, values[cut:])
            resumed = snapshot.copy()
            for x in resumed_values:
                resumed = step(resumed, x)
            np.testing.assert_array_equal(resumed, full)
            self.assertFalse(np.array_equal(snapshot, full))
            self.assertFalse(np.shares_memory(snapshot, full))

    def test_source_scope_and_npu_tracking_contract_unchanged(self):
        for path in (KDA, HYBRID, LEGACY, FB, COMPONENT, UNIFIED, POOL):
            self.assertEqual(text(path), text(path, True), path)
        current = node(SCHEDULE, "_mamba_radix_cache_v2_req_prepare_for_extend")
        baseline = node(SCHEDULE, "_mamba_radix_cache_v2_req_prepare_for_extend", True)
        found = []
        absolute_found = []
        expected_absolute = ast.parse(
            '''if getattr(self.tree_cache, "glm53_kpool_share_page_size", None) == checkpoint_grid:
    mamba_track_seqlen_aligned = (
        (len(req.prefix_indices) + req.extend_range.length)
        // checkpoint_grid * checkpoint_grid
    )
'''
        ).body[0]
        case = self

        class RemoveOptIn(ast.NodeTransformer):
            def visit_If(self, n):
                if FLAG in ast.unparse(n.test):
                    found.append(n)
                    return None
                if "glm53_kpool_share_page_size" in ast.unparse(n.test):
                    # Only this exact isolated assignment is allowed. A broader
                    # guard, else clause, or extra side effect must fail scope.
                    case.assertEqual(ast.dump(n), ast.dump(expected_absolute))
                    absolute_found.append(n)
                    return None
                return self.generic_visit(n)

        stripped = RemoveOptIn().visit(copy.deepcopy(current))
        self.assertEqual(len(found), 1)
        self.assertEqual(len(absolute_found), 1)
        self.assertEqual(ast.dump(stripped), ast.dump(baseline))
        before = ast.parse(text(SCHEDULE, True))
        after = ast.parse(text(SCHEDULE))
        bm = {
            n.name: ast.dump(n)
            for n in ast.walk(before)
            if isinstance(n, ast.FunctionDef)
        }
        am = {
            n.name: ast.dump(n)
            for n in ast.walk(after)
            if isinstance(n, ast.FunctionDef)
        }
        self.assertEqual(
            [k for k in bm if bm[k] != am[k]],
            ["_mamba_radix_cache_v2_req_prepare_for_extend"],
        )
        kda = ast.unparse(node(KDA, "forward_extend"))
        self.assertIn("track_lens = forward_batch.mamba_track_aligned_lens()", kda)
        self.assertIn("track_lens=track_lens", kda)
        transfer = text("python/sglang/srt/disaggregation/prefill.py")
        self.assertIn("req_index_to_mamba_index_mapping[", transfer)
        self.assertNotIn(FLAG, transfer)


if __name__ == "__main__":
    unittest.main()
