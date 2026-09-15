"""CPU regression for native GLM KPool prefix sharing and checkpoint depth.

Run with a local PyTorch interpreter:
    python -B test/manual/ascend/test_glm53_kpool_prefix_page_cpu.py -v

Extract actual source functions by AST; never import SGLang, torch_npu or
Triton. All tensors are CPU tensors. Synthetic per-group token vectors expose
address ownership, not model numerics. The radix dedup boundary is simulated;
the real pooled addressing and ReqToTokenPool.write methods perform the reads
and remap. This is not a PD/NPU/Mamba recurrence or quality test.
"""

import ast
import copy
import enum
import math
import os
import subprocess
import types
import unittest
from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
BASE = "b2d57c790e176334c73263a429613793489b69c9"
SCHEDULE = "python/sglang/srt/managers/schedule_batch.py"
BUILDER = "python/sglang/srt/mem_cache/kv_cache_builder.py"
INDEX = "python/sglang/srt/hardware_backend/npu/attention/glm53/kpool_indexer.py"
POOL = "python/sglang/srt/mem_cache/memory_pool.py"
FB = "python/sglang/srt/model_executor/forward_batch_info.py"
CONTEXT = "python/sglang/srt/runtime_context.py"
ARCH = "python/sglang/srt/configs/hybrid_arch.py"
FLAG = "SGLANG_GLM53_PD_PREFILL_REUSABLE_CHECKPOINT"
NS = types.SimpleNamespace
SOURCES = {p: (ROOT / p).read_text() for p in (SCHEDULE, BUILDER, INDEX, POOL, FB, CONTEXT, ARCH)}
BASE_SCHEDULE = subprocess.check_output(
    ["git", "-C", str(ROOT), "show", f"{BASE}:{SCHEDULE}"],
    text=True,
    env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
)


def extract(path, name, namespace, *, owner=None, baseline=False):
    tree = ast.parse(BASE_SCHEDULE if baseline else SOURCES[path])
    if owner is not None:
        owners = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == owner]
        assert len(owners) == 1, (path, owner)
        tree = owners[0]
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(matches) == 1, (path, name, len(matches))
    method = copy.deepcopy(matches[0])
    method.decorator_list = []
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        method,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace)
    return namespace[name]


class Mode(enum.Enum):
    EXTEND = 1
    DECODE = 2
    TARGET_VERIFY = 3


Entry = namedtuple("Entry", "track_mask track_index track_seqlen")
ENV = dict(
    os=os, math=math, torch=torch, ForwardMode=Mode,
    get_disagg=lambda: NS(disaggregation_mode="prefill"),
    get_exec=lambda: NS(mamba=NS(enable_mamba_extra_buffer=True, enable_mamba_extra_buffer_lazy=False)),
    get_parallel=lambda: NS(dp_size=1, pp_size=1, attn_cp_size=1, dcp_size=1),
    mamba_cache_chunk_size=lambda: 64,
    _MambaRadixCacheV2TrackEntry=Entry,
)
extract(CONTEXT, "mamba_checkpoint_grid", ENV)
PREPARE = extract(SCHEDULE, "_mamba_radix_cache_v2_req_prepare_for_extend", dict(ENV))
OLD_PREPARE = extract(SCHEDULE, "_mamba_radix_cache_v2_req_prepare_for_extend", dict(ENV), baseline=True)
MAX_PREFIX = extract(SCHEDULE, "_compute_max_prefix_len", {})
TRACK_LENS = extract(FB, "mamba_track_aligned_lens", dict(ENV))
ADDRESS = extract(INDEX, "_pooled_write_locs", {"torch": torch})
PAGE_TABLE = extract(INDEX, "_pooled_page_table", {"torch": torch})
ROW_WRITE = extract(POOL, "write", {"torch": torch}, owner="ReqToTokenPool")
BUILDER_ENV = {"math": math}
extract(ARCH, "glm5_next_config", BUILDER_ENV)
SHARE_PAGE = extract(BUILDER, "_glm53_npu_prefix_page_size", BUILDER_ENV)


def model(kpool=4, model_type="glm5_next", draft=False):
    text_cfg = NS(index_kpool=kpool)
    return NS(hf_config=NS(model_type=model_type, get_text_config=lambda: text_cfg), is_draft_model=draft)


def prepare(start, end, *, page=256, marker=True, final=False, flag=False, branch=None,
            baseline=False, device="npu:0", mode=Mode.EXTEND, logprob=-1):
    tree = NS(page_size=page, disable=False)
    if marker:
        tree.glm53_kpool_share_page_size = page
    req = NS(
        prefix_indices=range(start), origin_input_ids=range(end if final else end + 4096),
        output_ids=[], return_logprob=logprob >= 0, logprob_start_len=logprob,
        extend_range=NS(start=start, end=end, length=end-start),
        mamba_branching_seqlen=branch,
        kv=NS(mamba_ping_pong_track_buffer=torch.tensor([10, 11], device="cpu"),
              mamba_next_track_idx=0, mamba_last_track_idx=None, mamba_last_track_seqlen=None),
    )
    req._compute_max_prefix_len = types.MethodType(MAX_PREFIX, req)
    batch = NS(device=device, forward_mode=mode, tree_cache=tree,
               model_config=NS(hf_config=NS(architectures=["Glm5NextForConditionalGeneration"]),
                               hf_text_config=NS(mamba_chunk_size=64)),
               req_to_token_pool=NS(get_mamba_ping_pong_other_idx=lambda i: 1-i))
    with patch.dict(os.environ, {FLAG: "1" if flag else "0"}):
        entry = (OLD_PREPARE if baseline else PREPARE)(batch, req)
    return req, entry


def dedup_replay(lcp, share_page, *, length=4096):
    """Real CPU address helpers + real row write; simulated radix boundary."""
    assert length % 256 == 0 and 0 <= lcp <= length
    indexer = NS(index_kpool=4)
    n_pages = length // 64
    old_pages = torch.arange(1, n_pages + 1, device="cpu")
    new_pages = old_pages + n_pages
    groups = torch.arange(length // 4, device="cpu")
    old_tokens = torch.arange(length, device="cpu")
    new_tokens = old_tokens.clone()
    new_tokens[lcp:] += 1_000_000
    old_keys = old_tokens.view(-1, 4)
    new_keys = new_tokens.view(-1, 4)
    index = torch.full(((2*n_pages + 1)*64, 4), -1, dtype=torch.int64, device="cpu")
    index[ADDRESS(indexer, old_pages, groups)] = old_keys
    index[ADDRESS(indexer, new_pages, groups)] = new_keys
    old_row = (old_pages[:, None]*64 + torch.arange(64, device="cpu")).reshape(-1)
    new_row = (new_pages[:, None]*64 + torch.arange(64, device="cpu")).reshape(-1)
    pool = NS(req_to_token=new_row.view(1, -1).clone())
    shared = lcp // share_page * share_page
    ROW_WRITE(pool, (0, slice(0, shared)), old_row[:shared])
    table = pool.req_to_token[0, ::64] // 64
    locations = ADDRESS(indexer, table, groups)
    paged = PAGE_TABLE(indexer, table.unsqueeze(0))[0]
    # Independent entry-point consistency: pool-page table gives identical reads.
    table_locations = paged[groups // 64]*64 + groups % 64
    assert torch.equal(locations, table_locations)
    actual = index[locations]
    wrong = torch.nonzero((actual != new_keys).any(dim=1)).flatten().tolist()
    return wrong, table, old_pages, new_pages


class KPoolAddressTests(unittest.TestCase):
    def test_exact_3294_reproduces_nine_wrong_groups_and_256_fixes(self):
        wrong, table, old, new = dedup_replay(3294, 64)
        self.assertEqual(wrong, list(range(823, 832)))
        self.assertEqual(table[48], old[48])
        self.assertEqual(table[51], new[51])
        fixed, table, old, new = dedup_replay(3294, 256)
        self.assertEqual(fixed, [])
        self.assertEqual(table[47], old[47])
        self.assertEqual(table[48], new[48])
        self.assertEqual(table[51], new[51])

    def test_all_partial_anchor_residues_and_token_offsets(self):
        # All three page64 residues, every within-page token offset, at three anchors.
        for anchor in (0, 256, 3072):
            for residue in (64, 128, 192):
                for offset in range(64):
                    lcp = anchor + residue + offset
                    with self.subTest(lcp=lcp):
                        self.assertEqual(dedup_replay(lcp, 64)[0], list(range(lcp//4, (anchor+256)//4)))
                        self.assertEqual(dedup_replay(lcp, 256)[0], [])

    def test_aligned_boundaries_and_short_template_are_not_false_positives(self):
        for lcp in (0, 1, 55, 63, 256, 512, 3072, 3328, 4096):
            with self.subTest(lcp=lcp):
                for share in (64, 256):
                    self.assertEqual(dedup_replay(lcp, share)[0], [])

    def test_pooled_page_table_maps_only_anchors_and_clamps_padding(self):
        table = torch.tensor([[5, 91, 92, 93, 7, 94, 95, 96, -1, -1, -1, -1]], device="cpu")
        self.assertEqual(PAGE_TABLE(NS(index_kpool=4), table).tolist(), [[5, 7, 0]])


class PrefixPagePolicyTests(unittest.TestCase):
    def test_actual_model_classifier_and_lcm(self):
        self.assertEqual(SHARE_PAGE(model(), "npu:0", 64, 64, True), 256)
        self.assertEqual(SHARE_PAGE(model(), "npu", 64, 512, True), 512)
        self.assertEqual(SHARE_PAGE(model(1), "npu", 64, 64, True), 64)

    def test_noneligible_paths_preserve_original_page(self):
        for cfg, device, enabled in ((model(), "cpu", True), (model(), "cuda", True),
                                     (model(), "npu", False), (model(model_type="other"), "npu", True),
                                     (model(draft=True), "npu", True)):
            with self.subTest(device=device, enabled=enabled, cfg=cfg):
                self.assertEqual(SHARE_PAGE(cfg, device, 64, 64, enabled), 64)

    def test_invalid_kpool_and_physical_page_fail(self):
        for kpool in (0, -1, True, 4.0, "4", None):
            with self.subTest(kpool=kpool), self.assertRaises(ValueError):
                SHARE_PAGE(model(kpool), "npu", 64, 64, True)
        with self.assertRaises(ValueError):
            SHARE_PAGE(model(), "npu", 128, 64, True)


class CheckpointTests(unittest.TestCase):
    def assert_actual_snapshot_depth(self, req, entry, expected):
        self.assertTrue(entry.track_mask)
        self.assertEqual(req.kv.mamba_last_track_seqlen, expected)
        fb = NS(mamba_track_mask=torch.tensor([True], device="cpu"),
                mamba_track_seqlens=torch.tensor([entry.track_seqlen], device="cpu"),
                extend_prefix_lens=torch.tensor([len(req.prefix_indices)], device="cpu"))
        # Extracted ForwardBatch helper checks that the +1 sentinel names the same state.
        self.assertEqual(TRACK_LENS(fb).item() + len(req.prefix_indices), expected)
        self.assertGreater(expected, len(req.prefix_indices))
        self.assertLessEqual(expected, req.extend_range.end)

    def test_4032_continuation_absolute_depth_not_8128(self):
        req, entry = prepare(4032, 8128)
        self.assert_actual_snapshot_depth(req, entry, 7936)
        old_req, _ = prepare(4032, 8128, baseline=True)
        self.assertEqual(old_req.kv.mamba_last_track_seqlen, 8128)

    def test_all_physical_prefix_residues_and_long_tail_lengths(self):
        for prefix in (0, 64, 128, 192, 3072, 3136, 3200, 3264, 4032):
            for length in (256, 257, 319, 320, 511, 512, 4095, 4096, 4097):
                with self.subTest(prefix=prefix, length=length):
                    req, entry = prepare(prefix, prefix+length)
                    self.assert_actual_snapshot_depth(req, entry, (prefix+length)//256*256)

    def test_short_tail_does_not_donate_or_swap_slots(self):
        for prefix in (0, 64, 192, 4032):
            for length in (1, 63, 64, 127, 192, 255):
                with self.subTest(prefix=prefix, length=length):
                    req, entry = prepare(prefix, prefix+length, final=True, flag=True)
                    self.assertFalse(entry.track_mask)
                    self.assertEqual(entry.track_seqlen, -1)
                    self.assertIsNone(req.kv.mamba_last_track_seqlen)
                    self.assertIsNone(req.kv.mamba_last_track_idx)
                    self.assertEqual(req.kv.mamba_next_track_idx, 0)

    def test_exact_prompts_keep_reusable_checkpoint_on_256_grid(self):
        for end in (65536, 131072):
            req, entry = prepare(end-4096, end, final=True, flag=True)
            self.assert_actual_snapshot_depth(req, entry, end-256)
            self.assertLessEqual(req.kv.mamba_last_track_seqlen, req._compute_max_prefix_len(end))

    def test_no_final_opt_in_retains_end_checkpoint(self):
        for end in (65536, 131072):
            req, entry = prepare(end-4096, end, final=True, flag=False)
            self.assert_actual_snapshot_depth(req, entry, end)

    def test_branch_checkpoint_3072_overrides_chunk_end(self):
        req, entry = prepare(0, 4096, branch=3072)
        self.assert_actual_snapshot_depth(req, entry, 3072)

    def test_old64_behavior_matches_b2d_matrix(self):
        for prefix in (0, 64, 192, 4032):
            for length in (1, 63, 64, 65, 127, 128, 255, 256, 4095, 4096):
                for final in (False, True):
                    for flag in (False, True):
                        with self.subTest(prefix=prefix, length=length, final=final, flag=flag):
                            args = dict(page=64, marker=False, final=final, flag=flag)
                            a, ae = prepare(prefix, prefix+length, **args)
                            b, be = prepare(prefix, prefix+length, baseline=True, **args)
                            self.assertEqual(ae, be)
                            self.assertEqual(vars(a.kv) | {"mamba_ping_pong_track_buffer": None},
                                             vars(b.kv) | {"mamba_ping_pong_track_buffer": None})

    def test_unmarked_coarse_grid_preserves_old_behavior(self):
        a, ae = prepare(4032, 8128, marker=False)
        b, be = prepare(4032, 8128, marker=False, baseline=True)
        self.assertEqual(ae, be)
        self.assertEqual(a.kv.mamba_last_track_seqlen, b.kv.mamba_last_track_seqlen)

    def test_decode_and_verify_old64_behavior_remains_b2d(self):
        for mode in (Mode.DECODE, Mode.TARGET_VERIFY):
            for end in (4096, 65536, 131072):
                args = dict(page=64, marker=False, final=True, flag=True, mode=mode)
                a, ae = prepare(end-4096, end, **args)
                b, be = prepare(end-4096, end, baseline=True, **args)
                self.assertEqual(ae, be)
                self.assertEqual(a.kv.mamba_last_track_seqlen, b.kv.mamba_last_track_seqlen)

    def test_logprob_limit_does_not_force_ineligible_reusable_checkpoint(self):
        req, entry = prepare(0, 4096, final=True, flag=True, logprob=3000)
        self.assert_actual_snapshot_depth(req, entry, 4096)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
