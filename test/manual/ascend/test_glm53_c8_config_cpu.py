"""CPU checks for GLM5Next C8 dtype, construction, sizing and PD guards.

Run: CUDA_VISIBLE_DEVICES= python -B test/manual/ascend/test_glm53_c8_config_cpu.py -v
AST extraction executes production functions without importing the NPU server.
Kernel correctness is covered separately by the NPU packed-cache tests.
"""

from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import logging
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
SRT = "python/sglang/srt/"
CONFIG = SRT + "mem_cache/kv_cache_configurator.py"
DTYPE = SRT + "mem_cache/kv_cache_dtype.py"
SIZER = SRT + "model_executor/pool_configurator.py"
PREFIX = SRT + "disaggregation/glm53_decode_prefix.py"
ARCH = "Glm5NextForConditionalGeneration"
DSA_UTILS = "sglang.srt.layers.attention.dsa.utils"
NPU_UTILS = "sglang.srt.hardware_backend.npu.utils"
CP_UTILS = "sglang.srt.layers.cp.utils"
CONFIG_MODULE = "sglang.srt.mem_cache.kv_cache_configurator"


def extract(path, name, namespace, owner=None):
    tree = ast.parse((ROOT / path).read_text())
    if owner:
        tree = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner
        )
    node = copy.deepcopy(
        next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    )
    node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace
    )
    return namespace[name]


def model(arch=ARCH, **changes):
    hf = NS(architectures=[arch], index_topk=2048, index_head_dim=128)
    result = NS(
        hf_config=hf,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        index_head_dim=128,
        head_dim=512,
        context_len=8192,
        get_num_kv_heads=lambda *args: 1,
    )
    result.__dict__.update(changes)
    return result


def record_pool(*args, **kwargs):
    return NS(args=args, **kwargs)


class C8ConfigTests(unittest.TestCase):
    def setUp(self):
        self.parallel = NS(
            attn_cp_size=1, dcp_size=1, attn_dcp_size=1, attn_tp_size=2, pp_size=1
        )
        self.schedule = NS(
            page_size=64, max_total_tokens=532736, disable_overlap_schedule=True
        )
        self.spec = NS(
            speculative_algorithm="EAGLE",
            speculative_draft_kv_cache_dtype=None,
            speculative_draft_attention_backend="ascend",
        )
        self.memory = NS(
            enable_hisparse=False,
            enable_hierarchical_cache=False,
            enable_unified_cache_external_linker=False,
            disable_radix_cache=False,
        )
        self.execution = NS(
            kernel=NS(
                attention_backend="ascend",
                dsa_prefill_backend="ascend",
                dsa_decode_backend="ascend",
            ),
            features=NS(enable_memory_saver=False),
            mamba=NS(
                enable_mamba_extra_buffer=True, enable_mamba_extra_buffer_lazy=False
            ),
        )
        self.offload = False
        self.arch35 = False
        self.prefill_cp = False
        self.disagg = NS(disaggregation_mode="decode")
        self.model_args = NS(kv_cache_dtype="int8")
        self.env = dict(
            torch=torch,
            copy=copy.copy,
            _is_npu=True,
            _is_hip=False,
            logger=logging.getLogger(__name__),
            current_platform=NS(is_npu=lambda: True, is_cpu=lambda: False),
            fp8_dtype=torch.float8_e4m3fn,
            get_schedule=lambda: self.schedule,
            get_parallel=lambda: self.parallel,
            get_spec=lambda: self.spec,
            get_memory=lambda: self.memory,
            get_exec=lambda: self.execution,
            get_disagg=lambda: self.disagg,
            get_model=lambda: self.model_args,
            envs=NS(
                SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD=NS(get=lambda: self.offload),
                SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=NS(get=lambda: False),
            ),
            DSATokenToKVPool=NS(
                quant_block_size=128,
                rope_storage_dtype=torch.bfloat16,
                index_k_with_scale_buffer_dtype=torch.uint8,
            ),
            is_float4_e2m1fn_x2=lambda dtype: False,
            dsa_layer_skips_topk=lambda config, layer: False,
            mambaish_config=lambda config: None,
            HybridLinearKVPool=record_pool,
            MHATokenToKVPoolMXFP8=object,
            max_speculative_num_draft_tokens=lambda: 4,
            get_dsa_index_kpool=lambda config: 4,
            get_dsa_index_kpool_compress=lambda config: True,
            MemoryPoolConfig=lambda **kwargs: NS(**kwargs),
        )
        for name in (
            "_hf_arch",
            "_hf_attr",
            "is_deepseek_dsa",
            "get_dsa_index_head_dim",
        ):
            extract(SRT + "configs/model_config.py", name, self.env)
        for name in (
            "_validate_pd_int8_draft_dtype",
            "_validate_npu_int8_kv_cache",
            "calculate_mla_kv_cache_dim",
            "_should_elide_dsa_index_k",
        ):
            extract(CONFIG, name, self.env)
        self.resolve = extract(DTYPE, "configure_kv_cache_dtype", self.env)
        self.modules = {
            DSA_UTILS: NS(is_dsa_enable_prefill_cp=lambda: self.prefill_cp),
            NPU_UTILS: NS(is_npu_arch35=lambda: self.arch35),
            CP_UTILS: NS(
                get_glm_dsa_layer_split_effective_num_layers=lambda kvc, count: count,
                get_glm_dsa_cp_layer_shard_info=lambda kvc: (None, 1),
                get_layer_shard_range=lambda *args: None,
            ),
            CONFIG_MODULE: NS(**self.env),
            "sglang.srt.mem_cache.kv_cache_dtype": NS(
                configure_kv_cache_dtype=self.resolve
            ),
            "sglang.srt.hardware_backend.npu.memory_pool_npu": NS(
                NPUMLATokenToKVPool=record_pool
            ),
            "sglang.srt.hardware_backend.npu.attention.glm53.compact_index": NS(
                cache_options=lambda *args: {}
            ),
            "sglang.srt.layers.attention.qsa.config": NS(
                parse_qsa_profile=lambda config: None
            ),
            "sglang.srt.mem_cache.qsa_kv_pool": NS(QSATokenToKVPool=object),
        }
        self.patcher = patch.dict(sys.modules, self.modules)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def dtype(self, main="int8", **kwargs):
        args = dict(
            server_args_kv_cache_dtype=main,
            model=None,
            model_dtype=torch.bfloat16,
            is_draft_worker=False,
            is_dflash=False,
            speculative_draft_attention_backend="ascend",
        )
        args.update(kwargs)
        return self.resolve(**args)

    def width(self, config=None, dtype=torch.int8):
        return self.env["calculate_mla_kv_cache_dim"](
            model_config=config or model(), kv_cache_dtype=dtype
        )

    def kvc(self, dtype=torch.int8, draft=False):
        return NS(
            kv_cache_dtype=dtype,
            kv_cache_dtype_str="int8" if dtype == torch.int8 else "bf16",
            model_config=model(ARCH + "NextN" if draft else ARCH),
            model=None,
            model_dtype=torch.bfloat16,
            use_mla_backend=True,
            device="npu:0",
            pool_page_size=64,
            is_draft_worker=draft,
            ps=NS(pp_size=1),
            layer_info=NS(num_effective_layers=4, start_layer=0, end_layer=4),
            mambaish_config=None,
            server_args=NS(enable_hisparse=False),
            spec_algorithm=NS(
                is_eagle=lambda: True,
                is_standalone=lambda: False,
                is_dflash_family=lambda: False,
            ),
            spec_aux_config=NS(eagle_draft_num_layers=1),
            post_capture_kv_active=False,
            _build_mha_quant_method=lambda **kwargs: None,
        )

    def sizer(self, kvc):
        namespace = dict(self.env)
        extract(SIZER, "_get_dsa_cache_layer_ids", namespace)
        methods = {
            name: extract(SIZER, name, namespace, "DefaultPoolConfigurator")
            for name in (
                "__init__",
                "_compute_cell_size",
                "_compute_dsa_indexer_cell_size",
                "_compute_npu_c8_draft_cell_size",
                "calculate_pool_sizes",
            )
        }
        methods["_compute_qsa_cell_size"] = staticmethod(lambda **kwargs: 0)
        return type("Sizer", (), methods)(kvc)

    def test_cli_choices_and_defaults(self):
        parser = argparse.ArgumentParser()
        for path, name, expected in (
            ("model", "kv_cache_dtype", "auto"),
            ("spec", "speculative_draft_kv_cache_dtype", None),
        ):
            tree = ast.parse(
                (ROOT / (SRT + f"arg_groups/fields/{path}.py")).read_text()
            )
            field = next(
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.AnnAssign)
                and isinstance(n.target, ast.Name)
                and n.target.id == name
            )
            choices = next(
                ast.literal_eval(n.value)
                for n in ast.walk(field.annotation)
                if isinstance(n, ast.keyword) and n.arg == "choices"
            )
            self.assertEqual(ast.literal_eval(field.value), expected)
            parser.add_argument("--" + name.replace("_", "-"), choices=choices)
        parsed = parser.parse_args(
            ["--kv-cache-dtype", "int8", "--speculative-draft-kv-cache-dtype", "int8"]
        )
        self.assertEqual(
            (parsed.kv_cache_dtype, parsed.speculative_draft_kv_cache_dtype),
            ("int8", "int8"),
        )

    def test_dtype_target_draft_override_and_bf16_defaults(self):
        self.assertEqual(self.dtype(), (None, torch.int8))
        self.assertEqual(
            self.dtype(
                "bf16", is_draft_worker=True, speculative_draft_kv_cache_dtype="int8"
            ),
            ("int8", torch.int8),
        )
        self.assertEqual(self.dtype(is_draft_worker=True), (None, torch.int8))
        for name in ("auto", "bf16", "bfloat16"):
            self.assertEqual(self.dtype(name), (None, torch.bfloat16))
        self.assertEqual(self.dtype("fp8_e4m3"), (None, torch.float8_e4m3fn))
        self.assertEqual(self.dtype("fp8_e5m2"), (None, torch.float8_e5m2))

    def test_non_npu_and_fa4_reject_int8_instead_of_casting(self):
        self.env["_is_npu"] = False
        with self.assertRaisesRegex(ValueError, "only supported.*NPU"):
            self.dtype()
        self.env["_is_npu"] = True
        with self.assertRaisesRegex(ValueError, "ascend"):
            self.dtype(
                is_draft_worker=True,
                is_dflash=True,
                speculative_draft_attention_backend="fa4",
            )

    def test_cpu_tagged_torch_on_npu_uses_canonical_detector(self):
        # torch 2.10+cpu with torch_npu can report a CPU current_platform while
        # utils.is_npu() correctly identifies the active Ascend runtime.
        self.env["current_platform"].is_npu = lambda: False
        self.env["current_platform"].is_cpu = lambda: True
        self.env["_is_npu"] = True
        self.assertEqual(self.dtype(), (None, torch.int8))
        self.assertEqual(self.dtype(is_draft_worker=True), (None, torch.int8))
        self.assertEqual(
            self.dtype(
                "bf16",
                is_draft_worker=True,
                speculative_draft_kv_cache_dtype="int8",
            ),
            ("int8", torch.int8),
        )

    def test_exact_packed_width_includes_physical_zero_rope_for_target_and_nextn(self):
        for arch in (ARCH, ARCH + "NextN"):
            for mode in ("ascend", "trtllm"):
                self.execution.kernel.dsa_decode_backend = mode
                self.assertEqual(self.width(model(arch)), 512 + 128 + 16)
        self.assertEqual(self.width(dtype=torch.bfloat16), 512)
        self.execution.kernel.dsa_decode_backend = "ascend"
        self.assertEqual(self.width(dtype=torch.float8_e4m3fn), 528)
        self.assertEqual(
            self.width(model(qk_rope_head_dim=64), torch.float8_e4m3fn), 656
        )

    def test_unsupported_models_shapes_and_page_sizes_fail(self):
        bad_models = [
            model("OtherModel"),
            model("GlmMoeDsaForCausalLM"),
            model(kv_lora_rank=256),
            model(qk_rope_head_dim=64),
        ]
        bad_index = model()
        bad_index.hf_config.index_head_dim = 64
        bad_models.append(bad_index)
        no_dsa = model()
        no_dsa.hf_config.index_topk = None
        bad_models.append(no_dsa)
        for config in bad_models:
            with self.subTest(config=config), self.assertRaisesRegex(
                ValueError, "INT8 KV cache requires"
            ):
                self.width(config)
        self.schedule.page_size = 128
        with self.assertRaisesRegex(ValueError, "page64"):
            self.width()
        self.schedule.page_size = 64
        self.env["_is_npu"] = False
        with self.assertRaisesRegex(ValueError, "Ascend NPU"):
            self.width()

    def test_sparse_offload_and_all_cp_modes_fail_before_pool_creation(self):
        self.offload = True
        with self.assertRaisesRegex(ValueError, "sparse KV offload"):
            self.width()
        self.offload = False
        for attr in ("attn_cp_size", "dcp_size", "attn_dcp_size"):
            setattr(self.parallel, attr, 2)
            with self.subTest(attr=attr), self.assertRaisesRegex(
                ValueError, "CP attention kernel"
            ):
                self.width()
            setattr(self.parallel, attr, 1)
        self.prefill_cp = True
        with self.assertRaisesRegex(ValueError, "DSA prefill CP disabled"):
            self.width()
        # Existing BF16 and FP8 paths do not acquire the new C8 restriction.
        self.assertEqual(self.width(dtype=torch.bfloat16), 512)
        self.assertEqual(self.width(dtype=torch.float8_e4m3fn), 528)

    def test_configure_rejects_incompatible_backend_before_profiling(self):
        configure = extract(CONFIG, "configure", self.env, "KVCacheConfigurator")
        for change in ("mha", "other_backend", "cp"):
            kvc = self.kvc()
            kvc.use_mla_backend = change != "mha"
            self.execution.kernel.attention_backend = (
                "triton" if change == "other_backend" else "ascend"
            )
            self.parallel.attn_cp_size = 2 if change == "cp" else 1
            with self.subTest(change=change), self.assertRaisesRegex(
                ValueError, "INT8 KV cache requires"
            ):
                configure(kvc, pre_model_load_memory=0)

    def test_pd_mtp_dtype_guard_rejects_mixed_c8_on_both_workers(self):
        configure = extract(CONFIG, "configure", self.env, "KVCacheConfigurator")
        for mode in ("prefill", "decode"):
            self.disagg.disaggregation_mode = mode
            for target, draft in (("int8", "bf16"), ("int8", "bfloat16"),
                                  ("int8", "auto"), ("bf16", "int8"),
                                  ("auto", "int8")):
                self.model_args.kv_cache_dtype = target
                self.spec.speculative_draft_kv_cache_dtype = draft
                for is_draft in (False, True):
                    kvc = self.kvc(draft=is_draft)
                    # No memory/profile stubs: the guard must reject first.
                    with self.subTest(mode=mode, target=target, draft=draft,
                                      worker=is_draft), self.assertRaisesRegex(
                        ValueError, "PD MTP requires target and draft"
                    ):
                        configure(kvc, pre_model_load_memory=0)

    def test_pd_guard_preserves_inherited_dtype_and_non_pd_mixed_dtype(self):
        validate = self.env["_validate_pd_int8_draft_dtype"]
        for mode in ("prefill", "decode", "null"):
            self.disagg.disaggregation_mode = mode
            for target, draft in (("int8", None), ("int8", "int8"),
                                  ("bf16", None), ("bf16", "fp8_e4m3"),
                                  ("auto", "bfloat16")):
                self.model_args.kv_cache_dtype = target
                self.spec.speculative_draft_kv_cache_dtype = draft
                validate(spec_algorithm=NS(is_eagle=lambda: True))
            for target, draft in (("int8", "bf16"), ("bf16", "int8")):
                self.model_args.kv_cache_dtype = target
                self.spec.speculative_draft_kv_cache_dtype = draft
                validate(spec_algorithm=NS(is_eagle=lambda: False))
                if mode == "null":
                    validate(spec_algorithm=NS(is_eagle=lambda: True))

    def test_910b_native_pool_receives_int8_packed_width(self):
        build = extract(
            CONFIG, "_build_ascend_mla_kv_pool", self.env, "KVCacheConfigurator"
        )
        for draft in (False, True):
            result = build(
                self.kvc(draft=draft),
                max_total_num_tokens=532736,
                is_dsa_model=True,
                max_num_requests=2,
            )
            self.assertEqual(
                (result.dtype, result.kv_cache_dim, result.page_size),
                (torch.int8, 656, 64),
            )
            self.assertEqual(result.index_head_dim, 128)
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            result = build(
                self.kvc(dtype),
                max_total_num_tokens=64,
                is_dsa_model=True,
                max_num_requests=2,
            )
            self.assertIsNone(result.kv_cache_dim)
        self.arch35 = True
        result = build(
            self.kvc(torch.float8_e4m3fn),
            max_total_num_tokens=64,
            is_dsa_model=True,
            max_num_requests=2,
        )
        self.assertEqual(result.kv_cache_dim, 528)

    def test_hybrid_pool_preserves_fp32_state_and_bf16_default(self):
        build = extract(
            CONFIG, "_build_hybrid_linear_kv_pool", self.env, "KVCacheConfigurator"
        )
        state = NS(temporal=torch.zeros((1, 2, 2), dtype=torch.float32))
        req = NS(
            size=2,
            req_to_token=torch.zeros((2, 64), dtype=torch.int32),
            mamba_pool=state,
        )
        for dtype, width in ((torch.int8, 656), (torch.bfloat16, 512)):
            for draft in (False, True):
                kvc = self.kvc(dtype, draft)
                kvc.mambaish_config = NS(full_attention_layer_ids=[0, 2])
                result = build(
                    kvc,
                    max_total_num_tokens=64,
                    req_to_token_pool=req,
                    mha_pool_class=object,
                )
                self.assertEqual(result.kv_cache_dim, width)
                self.assertEqual(result.dtype, dtype)
                self.assertEqual(
                    result.full_attention_layer_ids, [0] if draft else [0, 2]
                )
                self.assertEqual(result.index_kpool, 4)
                self.assertIs(result.mamba_pool, state)
                self.assertEqual(result.mamba_pool.temporal.dtype, torch.float32)
        kvc.pool_page_size = 128
        kvc.kv_cache_dtype = torch.int8
        with self.assertRaisesRegex(ValueError, "page64"):
            build(
                kvc,
                max_total_num_tokens=64,
                req_to_token_pool=req,
                mha_pool_class=object,
            )

    def test_cell_size_and_capacity_charge_bf16_index_for_target_and_draft(self):
        for dtype, per_layer in (
            (torch.int8, 912),
            (torch.bfloat16, 1280),
            (torch.float8_e4m3fn, 660),
        ):
            for draft, layers in ((False, 5), (True, 4)):
                with self.subTest(dtype=dtype, draft=draft):
                    sized = self.sizer(self.kvc(dtype, draft))
                    self.assertEqual(sized._cell_size, layers * per_layer)
                    budget = 532736 * sized._cell_size + sized._cell_size - 1
                    self.assertEqual(
                        sized.calculate_pool_sizes(budget, 64).max_total_num_tokens,
                        532736,
                    )

    def test_mixed_target_draft_dtype_accounts_for_both_physical_layouts(self):
        cases = (
            (torch.bfloat16, "int8", 4 * 1408 + 912),
            (torch.int8, "bf16", 4 * 912 + 1408),
            (torch.int8, "bfloat16", 4 * 912 + 1408),
            (torch.int8, "auto", 4 * 912 + 1408),
            (torch.int8, "int8", 5 * 912),
        )
        for target_dtype, draft_dtype, expected in cases:
            with self.subTest(target=target_dtype, draft=draft_dtype):
                self.spec.speculative_draft_kv_cache_dtype = draft_dtype
                kvc = self.kvc(target_dtype)
                self.assertEqual(self.sizer(kvc)._cell_size, expected)
                self.assertFalse(kvc.is_draft_worker)
                self.assertEqual(kvc.kv_cache_dtype, target_dtype)

    def test_hicache_keeps_full_bf16_index_budget(self):
        self.memory.enable_hierarchical_cache = True
        self.assertEqual(self.sizer(self.kvc())._cell_size, 5 * 912)
        self.assertEqual(self.width(), 656)

    def test_decode_prefix_and_mamba_guards_are_dtype_independent(self):
        name = "sglang.srt.disaggregation.glm53_decode_prefix"
        spec = importlib.util.spec_from_file_location(name, ROOT / PREFIX)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {name: module}):
            spec.loader.exec_module(module)
            namespace = dict(
                self.env,
                resolving_view=lambda cfg: cfg,
                model_config_of=lambda cfg: model(),
                MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO=3,
                MAMBA_CACHE_BASE_RATIO_DROP_ON_SKIP=1,
                MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_OVERLAP=1,
            )
            ratio = extract(
                CONFIG, "_calculate_mamba_ratio", namespace, "KVCacheConfigurator"
            )
            extra = extract(
                SRT + "arg_groups/model_override_base.py",
                "mamba_extra_buffer_of",
                namespace,
            )
            for dtype in ("auto", "bf16", "int8"):
                cfg = NS(
                    device="npu",
                    disaggregation_mode="decode",
                    disaggregation_transfer_backend="ascend",
                    quantization="modelslim",
                    speculative_draft_model_quantization="modelslim",
                    tp_size=8,
                    dp_size=4,
                    ep_size=8,
                    pp_size=1,
                    dcp_size=1,
                    attn_cp_size=1,
                    nnodes=1,
                    enable_dp_attention=True,
                    page_size=64,
                    enable_hierarchical_cache=False,
                    disaggregation_decode_retraction_backup=None,
                    speculative_algorithm="EAGLE",
                    speculative_num_steps=3,
                    speculative_eagle_topk=1,
                    speculative_num_draft_tokens=4,
                    disaggregation_decode_enable_radix_cache=True,
                    disable_radix_cache=False,
                    mamba_radix_cache_strategy="extra_buffer",
                    kv_cache_dtype=dtype,
                    speculative_draft_kv_cache_dtype=dtype,
                )
                kvc = NS(server_args=cfg, model_config=model())
                profile = module.glm53_pd_decode_prefix_profile(cfg, kvc.model_config)
                self.assertEqual(profile.physical_page_size, 64)
                self.assertEqual(profile.prefix_share_page_size, 256)
                self.assertEqual(ratio(kvc), 1)
                self.assertFalse(extra(cfg))
                cfg.disaggregation_mode = "prefill"
                cfg.enable_hierarchical_cache = True
                self.assertEqual(ratio(kvc), 4)
                self.assertTrue(extra(cfg))


if __name__ == "__main__":
    unittest.main()
