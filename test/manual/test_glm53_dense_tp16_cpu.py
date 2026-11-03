"""Actual-source dense TP16 contracts; CPU tensors and mocked device boundaries.

Run with python -B, taskset -c 30,31, TORCH_DEVICE_BACKEND_AUTOLOAD=0 and
accelerator visibility empty. No application/backend modules are imported.
This checks loading/layout/bootstrap, not CANN quantization or HCCL liveness.
"""

from __future__ import annotations

import __future__
import ast
import builtins
import gc
import importlib.abc
import importlib.util
import itertools
import json
import logging
import os
from pathlib import Path
import sys
import time
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from dataclasses import dataclass, field
from types import MappingProxyType, SimpleNamespace as NS
import unittest
from unittest import mock
import weakref

assert set(os.sched_getaffinity(0)) <= {30, 31}, "Use CPU30/31"
assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert sys.dont_write_bytecode, "Use python -B"
for key in (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "ASCEND_RT_VISIBLE_DEVICES",
    "ASCEND_VISIBLE_DEVICES",
    "NPU_VISIBLE_DEVICES",
):
    assert os.environ.get(key, "") == "", key
    os.environ[key] = ""


class NoBackends(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in (
            "torch_npu",
            "sgl_kernel",
            "sgl_kernel_npu",
            "sglang",
        ):
            raise RuntimeError(f"CPU test forbids import {fullname}")


sys.meta_path.insert(0, NoBackends())
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
assert not torch.cuda.is_initialized()
SRT = Path(__file__).resolve().parents[2] / "python/sglang/srt"
HELPER = "sglang.srt.hardware_backend.npu.moe.glm53_dense_tp"
ENV = dict(
    SGLANG_GLM53_DENSE_TP16="1",
    SGLANG_GLM53_4NODE_PD="1",
    SGLANG_GLM53_NORMAL_HCCL="1",
    SGLANG_GLM53_NORMAL_SKIP_IDLE_DP="0",
)


def load(path, names, ns):
    """Compile intact class/function ASTs, preserving super() and decorators."""
    tree = ast.parse((SRT / path).read_text())
    nodes = []
    for name in names:
        body = tree.body
        for part in name.split("."):
            node = next(n for n in body if getattr(n, "name", None) == part)
            body = node.body
        nodes.append(node)
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]),
            str(SRT / path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        ns,
    )


def namespace():
    modules = {}

    def imports(name, globals=None, locals=None, fromlist=(), level=0):
        if name in modules:
            return modules[name]
        return builtins.__import__(name, globals, locals, fromlist, level)

    return dict(
        __builtins__=dict(vars(builtins), __import__=imports),
        __name__=__name__,
        torch=torch,
        nn=torch.nn,
        os=os,
        time=time,
        modules=modules,
        logger=logging.getLogger(__name__),
    )


def profile(**changes):
    result = dict(
        moe_dense_tp_size=16,
        tp_size=32,
        ep_size=32,
        dp_size=4,
        nnodes=4,
        pp_size=1,
        attn_cp_size=1,
        dcp_size=1,
        moe_dp_size=1,
        dwdp_size=1,
        enable_dp_attention=True,
        disaggregation_mode="prefill",
        quantization="modelslim",
        moe_a2a_backend="deepep",
        deepep_mode="normal",
        disable_overlap_schedule=True,
        disable_cuda_graph=True,
        enable_two_batch_overlap=False,
        enable_single_batch_overlap=False,
        enable_eplb=False,
        ep_num_redundant_experts=0,
        enable_prefill_cp=False,
        enable_layernorm_sp=False,
        enable_dense_mlp_attn_tp=False,
        enable_symm_mem=False,
        enable_torch_symm_mem=False,
        enable_quant_communications=False,
        enable_torch_compile=False,
        enable_pdmux=False,
        elastic_ep_backend=None,
        ep_join_mode=None,
        ep_join_rank_offset=0,
        weight_cache_mode="off",
        custom_weight_loader=[],
        load_format="safetensors",
        cuda_graph_config=NS(
            decode=NS(backend="disabled"), prefill=NS(backend="disabled")
        ),
    )
    result.update(changes)
    return NS(**result)


def helper_ns():
    ns = namespace()
    load(
        "arg_groups/model_override_base.py",
        ["ResolvingConfig", "resolving_view"],
        ns,
    )
    ns["modules"]["sglang.srt.arg_groups.model_override_base"] = NS(
        resolving_view=ns["resolving_view"]
    )
    load(
        "hardware_backend/npu/moe/glm53_dense_tp.py",
        ["dense_tp16_enabled", "validate_dense_tp16_model"],
        ns,
    )
    return ns


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(mock.patch.dict(os.environ, ENV))
        self.ns = helper_ns()

    def test_strict_profile_and_inactive_legacy_values(self):
        gate = self.ns["dense_tp16_enabled"]
        self.assertTrue(gate(profile(), is_npu=True))
        # Each supported constraint independently rejects an unsupported case.
        for key, value in vars(profile()).items():
            if key == "cuda_graph_config":
                bad = NS(decode=NS(backend="disabled"), prefill=NS(backend="full"))
            elif key == "load_format":
                bad = "presharded"
            elif isinstance(value, bool):
                bad = not value
            elif value is None:
                bad = "unsupported"
            elif isinstance(value, int):
                bad = value + 1
            else:
                bad = "unsupported"
            with self.subTest(key=key), self.assertRaises(ValueError):
                gate(profile(**{key: bad}), is_npu=True)
        for key in ENV:
            if key.endswith("SKIP_IDLE_DP"):
                continue
            with mock.patch.dict(os.environ, {key: "0"}), self.assertRaises(ValueError):
                gate(profile(), is_npu=True)
        with self.assertRaises(ValueError):
            gate(profile(), is_npu=False)
        for args in (
            profile(moe_dense_tp_size=None),
            profile(moe_dense_tp_size=1),
            profile(moe_dense_tp_size=32),
            profile(tp_size=16, moe_dense_tp_size=16),
        ):
            with mock.patch.dict(os.environ, {k: "0" for k in ENV}):
                self.assertFalse(gate(args, is_npu=False))

    def test_model_gate_and_actual_cli_validation(self):
        validate = self.ns["validate_dense_tp16_model"]
        text = NS(
            hidden_size=4096,
            intermediate_size=12288,
            first_k_dense_replace=3,
            moe_layer_freq=1,
            mhc=True,
            hidden_act="silu",
            swiglu_limit=10,
        )
        model = NS(
            hf_config=NS(architectures=["Glm5NextForConditionalGeneration"]),
            hf_text_config=text,
        )
        validate(model)
        for key in vars(text):
            changed = NS(**vars(text))
            setattr(changed, key, None)
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(NS(hf_config=model.hf_config, hf_text_config=changed))
        with self.assertRaises(ValueError):
            validate(
                NS(
                    hf_config=NS(
                        architectures=["Glm5NextForConditionalGenerationNextN"]
                    )
                )
            )
        ns = self.ns
        ns.update(
            resolving_view=lambda x: x,
            get_platform=lambda: NS(is_npu=True),
            is_runai_obj_uri=lambda _: False,
        )

        # Stop at LoRA validation, immediately after the real dense CLI gate.
        class AfterDenseGate(Exception):
            pass

        def stop(_):
            raise AfterDenseGate

        ns["modules"].update(
            {
                HELPER: NS(
                    **{
                        k: ns[k]
                        for k in ("dense_tp16_enabled", "validate_dense_tp16_model")
                    }
                ),
                "sglang.srt.arg_groups.lora_hook": NS(check_lora_server_args=stop),
            }
        )
        load("arg_groups/validation_hook.py", ["check_server_args"], ns)
        args = profile(
            pp_max_micro_batch_size=None,
            disable_cuda_graph_padding=False,
            min_free_slots_delay=None,
            base_gpu_id=0,
            gpu_id_step=1,
            served_model_name="glm",
        )
        with self.assertRaises(AfterDenseGate):
            ns["check_server_args"](args)
        args.moe_dense_tp_size = 8
        with self.assertRaises(ValueError):
            ns["check_server_args"](args)

    def test_saved_effective_args_types_and_helper_import_boundary(self):
        fixture = json.loads(
            (
                Path(__file__).parent / "fixtures/glm53_dense_tp16_effective_args.json"
            ).read_text()
        )
        args = fixture["effective_args"]
        self.assertEqual(args["custom_weight_loader"], [])
        self.assertIsNone(args["ep_join_mode"])
        graph_ns = namespace()
        graph_ns.update(dataclass=dataclass, field=field)
        load(
            "model_executor/cuda_graph_config.py",
            ["Phase", "Backend", "PhaseConfig", "CudaGraphConfig"],
            graph_ns,
        )
        phase = graph_ns["PhaseConfig"]
        args["cuda_graph_config"] = graph_ns["CudaGraphConfig"](
            **{key: phase(**value) for key, value in args["cuda_graph_config"].items()}
        )
        args["moe_dense_tp_size"] = 16  # only topology CLI change from saved TP1
        gate = self.ns["dense_tp16_enabled"]
        self.assertTrue(gate(NS(**args), is_npu=True))
        args["custom_weight_loader"] = None
        self.assertTrue(gate(NS(**args), is_npu=True))
        args["custom_weight_loader"] = ["custom.loader"]
        with self.assertRaises(ValueError):
            gate(NS(**args), is_npu=True)
        path = SRT / "hardware_backend/npu/moe/glm53_dense_tp.py"
        tree = ast.parse(path.read_text())
        self.assertEqual(
            [n.names[0].name for n in tree.body if isinstance(n, ast.Import)], ["os"]
        )
        self.assertFalse(any(isinstance(n, ast.ImportFrom) for n in tree.body))
        # These are namespace packages; no additional parent initializer executes.
        for parent in (
            SRT,
            SRT / "hardware_backend",
            SRT / "hardware_backend/npu",
            SRT / "hardware_backend/npu/moe",
        ):
            self.assertFalse((parent / "__init__.py").exists())
        before = set(sys.modules)
        spec = importlib.util.spec_from_file_location("dense_tp16_pure", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(set(sys.modules) - before, set())


class PG:
    def __init__(self, ranks, backend):
        self.ranks, self.backend, self.destroyed = tuple(ranks), backend, False


def parallel_ns(rank=0, world_size=32, actual_npu=False):
    ns = namespace()
    trace, refs = [], []

    def new_group(ranks, backend, **kw):
        trace.append((tuple(ranks), backend))
        obj = PG(ranks, backend)
        refs.append(weakref.ref(obj))
        return obj

    def destroy(pg):
        assert not pg.destroyed
        pg.destroyed = True

    dist = NS(
        is_initialized=lambda: True,
        get_rank=lambda: rank,
        get_world_size=lambda group=None: (
            world_size if group is None else len(group.ranks)
        ),
        get_backend=lambda _: "hccl",
        new_group=new_group,
        destroy_process_group=destroy,
    )

    class TorchProxy:
        distributed = dist
        device = staticmethod(lambda _: torch.device("cpu"))
        get_device_module = staticmethod(lambda _: NS())

        def __getattr__(self, key):
            return getattr(torch, key)

    ns.update(
        torch=TorchProxy(),
        dist=dist,
        _is_npu=True,
        _is_xpu=False,
        _is_musa=False,
        timedelta=timedelta,
        get_int_env_var=lambda *a: 0,
        is_cuda_alike=lambda: False,
        is_hip=lambda: False,
        is_cuda=lambda: False,
        _group_name_counter={},
        _groups={},
        weakref=weakref,
        _MODEL_PARALLEL_GROUP_TIMEOUT=None,
        get_torch_distributed_pg_options=lambda _: None,
        _ENABLE_CUSTOM_ALL_REDUCE=False,
        _ENABLE_MSCCLPP_ALL_REDUCE=False,
        _ENABLE_TORCH_SYMM_MEM_ALL_REDUCE=False,
        envs=NS(SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=NS(get=lambda: False)),
        get_parallel=lambda: NS(clear_derived_widths=lambda: None),
        get_global_dwdp_manager=lambda: None,
        set_global_dwdp_manager=lambda _: None,
        get_world_group=lambda: NS(local_rank=rank % 8, device_group=None),
        trace=trace,
        refs=refs,
    )
    load(
        "runtime_context.py", ["derive_attention_widths", "derive_parallel_widths"], ns
    )
    load("distributed/parallel_state.py", ["_get_unique_name", "_register_group"], ns)
    # Communicator construction/device transport are mocked, but group rank-loop,
    # helper options and destroy execute the actual production methods.
    disabled = lambda **kw: NS(disabled=True)

    def npu(group):
        return NS(disabled=False, group=group)

    if actual_npu:
        load(
            "distributed/device_communicators/npu_communicator.py",
            ["NpuCommunicator"],
            ns,
        )
        npu = ns["NpuCommunicator"]

    for module, values in {
        "custom_all_reduce": dict(dispatch_custom_allreduce=disabled),
        "pymscclpp": dict(PyMscclppCommunicator=disabled),
        "pynccl": dict(PyNcclCommunicator=disabled),
        "pynccl_allocator": dict(
            debug_check_symmetric_mempool=lambda *a: None,
            is_symmetric_memory_enabled=lambda: False,
            use_symmetric_memory=lambda *a, **kw: nullcontext(),
        ),
        "torch_symm_mem": dict(TorchSymmMemCommunicator=disabled),
        "hpu_communicator": dict(HpuCommunicator=disabled),
        "xpu_communicator": dict(XpuCommunicator=disabled),
        "npu_communicator": dict(NpuCommunicator=npu),
        "shm_broadcast": dict(MessageQueue=NS()),
    }.items():
        ns["modules"]["sglang.srt.distributed.device_communicators." + module] = NS(
            **values
        )
    ns["modules"]["sglang.srt.layers.dp_attention"] = NS(
        is_allocation_symmetric=lambda: False
    )
    ns["modules"]["sglang.srt.layers.sampler"] = NS(SYNC_TOKEN_IDS_ACROSS_TP=False)
    # Compile actual methods; only device allocation/transport are replaced.
    ctor_ns = dict(ns)
    load(
        "distributed/parallel_state.py",
        [
            "GroupCoordinator.__init__",
            "GroupCoordinator.destroy",
            "GroupCoordinator.all_reduce",
        ],
        ctor_ns,
    )
    ns["GroupCoordinator"] = type(
        "GroupCoordinator",
        (),
        {name: ctor_ns[name] for name in ("__init__", "destroy", "all_reduce")},
    )
    for name in (
        "_TP",
        "_DENSE_TP",
        "_PP",
        "_DCP",
        "_ATTN_CP",
        "_ATTN_TP",
        "_MOE_DP",
        "_MOE_EP",
        "_MOE_TP",
        "_SELF_PP",
        "_PDMUX_PREFILL_TP_GROUP",
    ):
        ns[name] = None
    load(
        "distributed/parallel_state.py",
        [
            "init_model_parallel_group",
            "initialize_model_parallel",
            "get_dense_tp_group",
            "destroy_model_parallel",
        ],
        ns,
    )
    return ns


def initialize(ns, dense=16, **changes):
    kw = dict(
        tensor_model_parallel_size=32,
        expert_model_parallel_size=32,
        attention_data_parallel_size=4,
        backend="hccl",
        dense_tensor_model_parallel_size=dense,
    )
    kw.update(changes)
    ns["initialize_model_parallel"](**kw)


class GroupTests(unittest.TestCase):
    def test_actual_npu_dispatch_preserves_dtype_and_global_groups(self):
        # Fake device-tagged payloads contain CPU tensors. Actual coordinator and
        # NpuCommunicator execute; the c10d boundary records the selected group.
        for rank in (0, 15, 16, 31):
            ns = parallel_ns(rank, actual_npu=True)
            initialize(ns)
            group = ns["get_dense_tp_group"]()
            global_groups = {name: ns[name] for name in ("_TP", "_ATTN_TP", "_MOE_EP")}
            self.assertEqual(
                [g.world_size for g in global_groups.values()], [32, 8, 32]
            )
            calls = []

            def reduce(payload, *, group):
                calls.append((tuple(group.ranks), payload.data.dtype))
                payload.data.copy_(
                    sum(torch.full_like(payload.data, r % 16 + 1) for r in group.ranks)
                )

            ns["dist"].all_reduce = reduce
            payload = NS(is_cpu=False, data=torch.zeros(2, 3, dtype=torch.bfloat16))
            result = group.all_reduce(payload)
            self.assertIs(result, payload)
            self.assertTrue(torch.equal(result.data, torch.full_like(result.data, 136)))
            self.assertEqual(
                calls,
                [(tuple(range(rank // 16 * 16, rank // 16 * 16 + 16)), torch.bfloat16)],
            )
            self.assertTrue(all(ns[name] is obj for name, obj in global_groups.items()))
            self.assertIsNot(group.device_group, ns["_TP"].device_group)
            device, cpu = group.device_group, group.cpu_group
            ns["destroy_model_parallel"]()
            self.assertTrue(device.destroyed and cpu.destroyed)
            self.assertIsNone(ns["get_dense_tp_group"]())

    def test_actual_all_rank_creation_order_and_repeated_destroy(self):
        traces = []
        for rank in range(32):
            ns = parallel_ns(rank)
            initialize(ns)
            group = ns["get_dense_tp_group"]()
            self.assertEqual(group.world_size, 16)
            self.assertEqual(group.rank_in_group, rank % 16)
            self.assertEqual(
                group.ranks, list(range(rank // 16 * 16, rank // 16 * 16 + 16))
            )
            self.assertFalse(
                group.use_pynccl
                or group.use_custom_allreduce
                or group.use_pymscclpp
                or group.use_torch_symm_mem_all_reduce
                or group.use_message_queue_broadcaster
            )
            traces.append(ns["trace"][:])
            self.assertEqual(
                ns["trace"][-4:],
                [
                    (tuple(range(16)), "hccl"),
                    (tuple(range(16)), "gloo"),
                    (tuple(range(16, 32)), "hccl"),
                    (tuple(range(16, 32)), "gloo"),
                ],
            )
            device, cpu = group.device_group, group.cpu_group
            group_ref = weakref.ref(group)
            registered = ns["_groups"][group.unique_name]
            ns["destroy_model_parallel"]()
            self.assertTrue(device.destroyed and cpu.destroyed)
            self.assertIsNone(ns["get_dense_tp_group"]())
            ns["destroy_model_parallel"]()  # New group destruction is idempotent.
            del group
            gc.collect()
            self.assertIsNone(group_ref())
            self.assertIsNone(registered())
            initialize(ns)
            ns["destroy_model_parallel"]()
        self.assertTrue(all(trace == traces[0] for trace in traces))

    def test_default_no_new_groups_and_unsupported_shapes_reject_early(self):
        base = parallel_ns()
        initialize(base, dense=None)
        self.assertIsNone(base["get_dense_tp_group"]())
        self.assertNotIn((tuple(range(16)), "hccl"), base["trace"])
        baseline_calls = len(base["trace"])
        candidate = parallel_ns()
        initialize(candidate)
        self.assertEqual(len(candidate["trace"]), baseline_calls + 4)
        decode = parallel_ns(world_size=16)
        initialize(
            decode,
            dense=None,
            tensor_model_parallel_size=16,
            expert_model_parallel_size=16,
            attention_data_parallel_size=8,
        )
        self.assertIsNone(decode["get_dense_tp_group"]())
        self.assertFalse(any(len(ranks) == 16 for ranks, _ in decode["trace"][-2:]))
        for changes in (
            dict(dense=8),
            dict(backend="gloo"),
            dict(duplicate_tp_group=True),
            dict(enable_symm_mem=True),
            dict(rank_offset=1),
            dict(attention_data_parallel_size=2),
            dict(max_world_size=32),
        ):
            ns = parallel_ns()
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                initialize(ns, **changes)
            self.assertEqual(ns["trace"], [])


def linear_ns():
    ns = namespace()
    ns.update(
        Parameter=torch.nn.Parameter,
        UninitializedParameter=torch.nn.UninitializedParameter,
        _is_cpu=False,
        _is_npu=True,
        _is_hip=False,
        _disable_hip_linear_quant=False,
        WEIGHT_LOADER_V2_SUPPORTED=[],
        itertools=itertools,
        wrap_method_with_debug_kernel_once=lambda *a, **kw: None,
        divide=lambda n, d: n // d,
        add_prefix=lambda name, prefix: prefix + "." + name if prefix else name,
        _use_aiter=False,
        envs=NS(SGLANG_NPU_W4A4_NEW_PACKING=NS(get=lambda: False)),
        get_parallel=lambda: NS(tp_rank=0, tp_size=32),
        get_forward=lambda: NS(sp_active=False),
        should_skip_mlp_all_reduce=lambda: False,
        get_tp_group=lambda: NS(),
        use_symmetric_memory=lambda *a, **kw: nullcontext(),
        is_allocation_symmetric=lambda: False,
        tensor_model_parallel_all_reduce=mock.Mock(
            side_effect=AssertionError("global TP reduction")
        ),
    )
    load("utils/common.py", ["set_weight_attrs"], ns)
    load("model_loader/weight_utils.py", ["default_weight_loader"], ns)
    load(
        "layers/parameter.py",
        [
            "BasevLLMParameter",
            "_ColumnvLLMParameter",
            "RowvLLMParameter",
            "ModelWeightParameter",
            "ChannelQuantScaleParameter",
        ],
        ns,
    )
    load(
        "hardware_backend/npu/quantization/linear_method_npu.py",
        ["NPUW8A8Int8DynamicLinearMethod.apply"],
        ns,
    )
    dynamic_apply = ns.pop("apply")
    ns["NPUW8A8Int8DynamicLinearMethod"] = type(
        "NPUW8A8Int8DynamicLinearMethod", (), {"apply": dynamic_apply}
    )
    ns["ModelSlimLinearScheme"] = type(
        "ModelSlimLinearScheme", (), {"requires_weight_loader_v2": False}
    )
    load(
        "layers/quantization/modelslim/schemes/modelslim_w8a8_int8.py",
        ["ModelSlimW8A8Int8"],
        ns,
    )
    ns["_NPULinearMethodBase"] = object
    load("layers/quantization/modelslim/modelslim.py", ["ModelSlimLinearMethod"], ns)
    load(
        "layers/linear.py",
        [
            "LinearBase",
            "ColumnParallelLinear",
            "MergedColumnParallelLinear",
            "RowParallelLinear",
        ],
        ns,
    )
    ns["SiluAndMul"] = (
        lambda: lambda x: torch.nn.functional.silu(x.chunk(2, -1)[0])
        * x.chunk(2, -1)[1]
    )
    load("models/deepseek_v2.py", ["DeepseekV2MLP"], ns)
    ns.update(
        QuantizationConfig=object,
        MappingProxyType=MappingProxyType,
        logger=NS(info_once=lambda *a: None, warning=lambda *a: None),
    )
    for name in (
        "ModelSlimW4A4Int4",
        "ModelSlimMXFP8Scheme",
        "ModelSlimMXFP4W4A8Scheme",
        "ModelSlimMXFP4Scheme",
    ):
        ns[name] = type(name, (), {})
    ns["modules"].update(
        {
            "sglang.srt.layers.linear": NS(LinearBase=ns["LinearBase"]),
            "sglang.srt.layers.moe.fused_moe_triton": NS(
                FusedMoE=type("FusedMoE", (), {})
            ),
        }
    )
    load("layers/quantization/modelslim/modelslim.py", ["ModelSlimConfig"], ns)

    class Quant(ns["ModelSlimConfig"]):
        def __init__(self):
            # Only config data is synthetic. Factory, scheme selection, prefix
            # resolution and layer.scheme assignment all execute actual source.
            self.packed_modules_mapping = {
                "model": {"gate_up_proj": ["gate_proj", "up_proj"]}
            }
            bases = ["mlp"] + [f"model.language_model.layers.{i}.mlp" for i in range(3)]
            self.quant_description = {
                f"{base}.{proj}.weight": "W8A8_DYNAMIC"
                for base in bases
                for proj in ("gate_proj", "up_proj", "down_proj")
            }
            self.quant_description[".weight"] = "W8A8_DYNAMIC"

    ns["Quant"] = Quant
    return ns


class WeightTests(unittest.TestCase):
    def test_historical_default_tp16_and_candidate_down_quant_domain(self):
        ns = linear_ns()
        parallel = NS(tp_rank=15, tp_size=16)
        ns["get_parallel"] = lambda: parallel
        ns["get_tp_group"] = lambda: NS(world_size=parallel.tp_size)
        old = ns["DeepseekV2MLP"](
            6, 12288, "silu", quant_config=ns["Quant"](), prefix="mlp"
        )
        parallel.tp_rank, parallel.tp_size = 31, 32
        candidate = ns["DeepseekV2MLP"](
            6,
            12288,
            "silu",
            quant_config=ns["Quant"](),
            prefix="mlp",
            tp_rank=15,
            tp_size=16,
            reduce_results=False,
        )
        current32 = ns["DeepseekV2MLP"](
            6, 12288, "silu", quant_config=ns["Quant"](), prefix="mlp"
        )
        self.assertEqual(old.gate_up_proj.output_partition_sizes, [768, 768])
        self.assertEqual(candidate.gate_up_proj.output_partition_sizes, [768, 768])
        self.assertEqual(current32.gate_up_proj.output_partition_sizes, [384, 384])
        load(
            "hardware_backend/npu/quantization/linear_method_npu.py",
            ["NPUW8A8Int8DynamicLinearMethod.process_weights_after_loading"],
            ns,
        )
        ns["NPUW8A8Int8DynamicLinearMethod"].process_weights_after_loading = ns[
            "process_weights_after_loading"
        ]
        prepared, captured = [], []

        def format_boundary(weight):
            prepared.append((tuple(weight.shape), weight.dtype))
            return weight  # Inspect actual pre-NZ preparation; no NZ execution.

        def quant_boundary(x):
            captured.append(x.detach().clone())
            # No emulated numerical-accuracy assertion: record exactly what the
            # actual dynamic method submits to the unavailable device operator.
            return torch.zeros_like(x, dtype=torch.int8), torch.ones(x.shape[0])

        class TorchProxy:
            ops = NS(
                npu=NS(
                    npu_dynamic_quant=quant_boundary,
                    npu_quant_matmul=lambda q, w, s, **kw: torch.zeros(
                        q.shape[0], w.shape[1], dtype=kw["output_dtype"]
                    ),
                )
            )

            def __getattr__(self, key):
                return getattr(torch, key)

        ns.update(torch=TorchProxy(), npu_format_cast=format_boundary)
        checkpoint = (torch.arange(6 * 12288).reshape(6, 12288) % 83).to(torch.int8)
        scale = torch.arange(1, 7, dtype=torch.float32).view(6, 1)
        features = (
            torch.arange(2 * 12288, dtype=torch.float32)
            .view(2, 12288)
            .to(torch.bfloat16)
        )
        for mlp in (old, candidate, current32):
            row = mlp.down_proj
            row.weight_loader(row.weight, checkpoint)
            row.weight_loader(row.weight_scale, scale)
            row.weight_loader(row.weight_offset, torch.zeros_like(scale))
            row.quant_method.process_weights_after_loading(row)
            width = row.input_size_per_partition
            start = row.tp_rank * width
            local = features[:, start : start + width].contiguous()
            # The historical projection normally reduces globally; stop that
            # reduction here to inspect its unchanged local quantization input.
            output, _ = row(local, skip_all_reduce=mlp is not candidate)
            self.assertEqual(output.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(row.weight_scale, scale.flatten()))
        self.assertEqual(
            prepared,
            [((768, 6), torch.int8), ((768, 6), torch.int8), ((384, 6), torch.int8)],
        )
        self.assertEqual([x.shape[-1] for x in captured], [768, 768, 384])
        self.assertTrue(torch.equal(old.down_proj.weight, candidate.down_proj.weight))
        self.assertTrue(torch.equal(captured[0], captured[1]))
        self.assertTrue(torch.equal(captured[1], features[:, 11520:12288]))
        self.assertTrue(torch.equal(captured[2], features[:, 11904:12288]))
        ns["tensor_model_parallel_all_reduce"].assert_not_called()

    def test_actual_dense_mapping_loader_scales_all_ranks_and_layers(self):
        ns = linear_ns()
        ns.update(
            FusedMoE=NS(make_expert_params_mapping=lambda **kw: []),
            DeepseekV2WeightLoaderMixin=NS(post_load_weights=lambda *a, **kw: None),
        )
        load(
            "models/glm5_next.py", ["Glm5NextForConditionalGeneration.load_weights"], ns
        )
        H, I = 6, 64  # local K4 vs TP32 K2; no full-size model allocation
        all_loaded = {}
        for rank in range(32):
            model = torch.nn.Module()
            model.model = torch.nn.Module()
            model.model.layers = torch.nn.ModuleList()
            model.config = NS(n_routed_experts=288)
            model.num_fused_shared_experts = 0
            model.quant_config = ns["Quant"]()
            weights = []
            oracle = {}
            for layer in range(3):
                block = torch.nn.Module()
                block.mlp = ns["DeepseekV2MLP"](
                    H,
                    I,
                    "silu",
                    quant_config=model.quant_config,
                    tp_rank=rank % 16,
                    tp_size=16,
                    reduce_results=False,
                    prefix=f"model.layers.{layer}.mlp",
                )
                model.model.layers.append(block)
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    shape = (H, I) if proj == "down_proj" else (I, H)
                    for field in ("weight", "weight_scale", "weight_offset"):
                        dim = shape if field == "weight" else (shape[0], 1)
                        values = torch.arange(torch.tensor(dim).prod().item()).reshape(
                            dim
                        )
                        values = (
                            (values + layer * 11 + (17 if proj == "up_proj" else 0))
                            % 83
                        ).to(torch.int8 if field == "weight" else torch.float32)
                        # Noncontiguous, nonzero-offset checkpoint views.
                        backing = torch.empty((*dim, 3), dtype=values.dtype)
                        backing[..., 1].copy_(values)
                        values = backing[..., 1]
                        name = f"model.language_model.layers.{layer}.mlp.{proj}.{field}"
                        weights.append((name, values))
                        oracle[layer, proj, field] = values.clone()
            with torch.no_grad():
                ns["load_weights"](model, iter(reversed(weights)))
            result = []
            j = rank % 16
            start, stop = j * (I // 16), (j + 1) * (I // 16)
            for layer, block in enumerate(model.model.layers):
                for field in ("weight", "weight_scale", "weight_offset"):
                    gate = oracle[layer, "gate_proj", field][start:stop]
                    up = oracle[layer, "up_proj", field][start:stop]
                    expected = torch.cat((gate, up))
                    actual = getattr(block.mlp.gate_up_proj, field)
                    self.assertTrue(torch.equal(actual, expected))
                    down = oracle[layer, "down_proj", field]
                    expected_down = down[:, start:stop] if field == "weight" else down
                    actual_down = getattr(block.mlp.down_proj, field)
                    self.assertTrue(torch.equal(actual_down, expected_down))
                    result.extend(
                        (actual.detach().clone(), actual_down.detach().clone())
                    )
                self.assertFalse(block.mlp.down_proj.reduce_results)
                self.assertIsNone(block.mlp.down_proj.bias)
                self.assertTrue(block.mlp.down_proj.scheme.is_dynamic)
                self.assertFalse(hasattr(block.mlp.down_proj.quant_method, "scheme"))
            all_loaded[rank] = result
            # Check gate/up reversal would be observable with tagged checkpoint.
            self.assertFalse(torch.equal(result[0], result[0].roll(I // 16, 0)))
        for rank in range(16):
            self.assertTrue(
                all(
                    torch.equal(a, b)
                    for a, b in zip(all_loaded[rank], all_loaded[rank + 16])
                )
            )

    def test_actual_dynamic_apply_uses_local_k_and_row_skips_global_reduce(self):
        ns = linear_ns()
        widths = []
        qmm = []

        def quant(x):
            widths.append(x.shape[-1])
            return x.to(torch.int8), torch.ones(x.shape[0])

        def matmul(q, w, scale, **kw):
            qmm.append(kw)
            return ((q.int() @ w.int()).float() * scale.flatten()).to(
                kw["output_dtype"]
            )

        class TorchProxy:
            ops = NS(npu=NS(npu_dynamic_quant=quant, npu_quant_matmul=matmul))

            def __getattr__(self, key):
                return getattr(torch, key)

        ns["torch"] = TorchProxy()
        row = ns["RowParallelLinear"](
            64,
            6,
            bias=False,
            tp_rank=3,
            tp_size=16,
            reduce_results=False,
            quant_config=ns["Quant"](),
        )
        row.weight.data.fill_(1)
        row.weight.data = (
            row.weight.data.t().contiguous()
        )  # CPU pre-NZ orientation only
        row.weight_scale.data.fill_(2)
        out, bias = row(torch.ones(3, 4))
        self.assertTrue(torch.equal(out, torch.full((3, 6), 8.0)))
        self.assertEqual(widths, [4])
        self.assertIsNone(bias)
        self.assertIsNone(qmm[0]["bias"])
        ns["tensor_model_parallel_all_reduce"].assert_not_called()


class BootstrapTests(unittest.TestCase):
    def test_actual_bootstrap_prewarm_before_memory_snapshot_and_draft_noop(self):
        ns = helper_ns()
        events = []
        live = []
        group = [None]
        free = [100.0]
        args = profile(
            max_ep_size=None,
            enable_p2p_check=True,
            dist_timeout=10,
            gated_launch_port=None,
        )
        exec_cfg = NS(
            moe=NS(
                is_ep_joiner=False, is_ep_scale_joiner=False, moe_a2a_backend="deepep"
            ),
            comm=NS(pre_warm_nccl=False, enable_symm_mem=False),
        )
        text = NS(
            hidden_size=4096,
            intermediate_size=12288,
            first_k_dense_replace=3,
            moe_layer_freq=1,
            mhc=True,
            hidden_act="silu",
            swiglu_limit=10,
        )
        model = NS(
            hf_config=NS(architectures=["Glm5NextForConditionalGeneration"]),
            hf_text_config=text,
        )

        def initialize_groups(**kw):
            events.append(("initialize", kw["dense_tensor_model_parallel_size"]))
            if kw["dense_tensor_model_parallel_size"] is not None:
                group[0] = NS(all_reduce=warm)

        def warm(tensor):
            events.append("warm")
            free[0] -= 3
            live.append(weakref.ref(tensor))
            return tensor

        def memory(*a, **kw):
            events.append(("memory", kw.get("distributed", False)))
            return free[0]

        class TorchProxy:
            def zeros(self, *a, device=None, **kw):
                self_device = device
                assert self_device == "npu:0"
                return torch.zeros(*a, **kw)

        ns.update(
            torch=TorchProxy(),
            get_parallel=lambda: args,
            get_exec=lambda: exec_cfg,
            get_serving=lambda: NS(host="local"),
            get_disagg=lambda: args,
            is_npu=lambda: True,
            get_dense_tp_group=lambda: group[0],
            get_available_gpu_memory=memory,
            _resolve_backend=lambda **kw: "hccl",
            _resolve_dist_init_method=lambda **kw: "local",
            _set_all_reduce_flags=lambda: None,
            init_distributed_environment=lambda **kw: events.append("world"),
            initialize_model_parallel=initialize_groups,
            _tag_groups_for_flashinfer_allreduce_only=lambda: None,
            initialize_dp_attention=lambda **kw: None,
            initialize_layernorm_sp=lambda **kw: None,
            register_sgl_tp_rank=lambda _: None,
            current_platform=NS(
                synchronize=lambda: events.append("sync"),
                empty_cache=lambda: events.append("empty_cache"),
            ),
            maybe_wait_for_gated_launch=lambda **kw: None,
            get_world_group=lambda: NS(world_size=32, cpu_group=None),
            get_tp_group=lambda: NS(),
            get_pp_group=lambda: NS(),
            _check_tp_memory_balance=lambda **kw: None,
            TorchDistributedResult=lambda **kw: NS(**kw),
        )
        args.attn_tp_group = NS()
        ns["modules"][HELPER] = NS(
            **{k: ns[k] for k in ("dense_tp16_enabled", "validate_dense_tp16_model")}
        )
        load(
            "distributed/bootstrap.py",
            ["_init_parallel_groups", "init_torch_distributed"],
            ns,
        )
        ps = NS(
            gpu_id=0,
            tp_size=32,
            tp_rank=0,
            pp_size=1,
            pp_rank=0,
            attn_dp_size=4,
            attn_cp_size=1,
            moe_ep_size=32,
            moe_dp_size=1,
            attn_dcp_size=1,
        )
        with mock.patch.dict(os.environ, ENV):
            result = ns["init_torch_distributed"](
                server_args=args,
                model_config=model,
                device="npu",
                ps=ps,
                dist_port=0,
                is_draft_worker=False,
                local_omp_cpuid=None,
            )
        self.assertEqual(result.pre_model_load_memory, 97.0)
        self.assertLess(events.index("warm"), events.index(("memory", True)))
        self.assertLess(events.index("empty_cache"), events.index(("memory", True)))
        self.assertEqual(events.count("warm"), 1)
        self.assertIsNone(live[0](), "warmup tensor must not survive startup")
        events.clear()
        ns["init_torch_distributed"](
            server_args=args,
            model_config=model,
            device="npu",
            ps=ps,
            dist_port=0,
            is_draft_worker=True,
            local_omp_cpuid=None,
        )
        self.assertNotIn("world", events)
        self.assertNotIn("warm", events)
        for dense in (None, 1, 32):
            args.moe_dense_tp_size = dense
            group[0] = None
            events.clear()
            ns["init_torch_distributed"](
                server_args=args,
                model_config=model,
                device="npu",
                ps=ps,
                dist_port=0,
                is_draft_worker=False,
                local_omp_cpuid=None,
            )
            self.assertIn(("initialize", None), events)
            self.assertNotIn("warm", events)


def decoder_fixture(rank, dense=16, group=None, tp=32):
    ns = linear_ns()
    parallel = NS(tp_rank=rank, tp_size=tp, moe_dense_tp_size=dense)

    class Forward:
        sp_active = False

        @contextmanager
        def scoped(self, **kw):
            yield

    forward = Forward()
    ns.update(
        get_parallel=lambda: parallel,
        get_forward=lambda: forward,
        get_spec=lambda: NS(speculative_algorithm=None),
        SpeculativeAlgorithm=NS(from_string=lambda _: None),
        get_dense_tp_group=lambda: group,
        enable_moe_dense_fully_dp=lambda: parallel.moe_dense_tp_size == 1,
        envs=NS(SGLANG_USE_AG_AFTER_QLORA=NS(get=lambda: False)),
        LayerScatterModes=NS(init_new=lambda **kw: NS(mlp_mode="FULL")),
        Glm5NextMLP=ns["DeepseekV2MLP"],
        nullcontext=nullcontext,
        get_attn_tp_context=lambda: NS(clear_attn_inputs=lambda: None),
    )

    class Attention(torch.nn.Module):
        def __init__(self, **kw):
            super().__init__()

        prepare_qkv_latent = None

        def forward(self, **kw):
            return kw["hidden_states"]

    class Moe(Attention):
        def __init__(self, **kw):
            super().__init__()
            self.experts = NS(moe_runner_config=NS(inplace=True))

    class Communicator:
        def __init__(self, **kw):
            self.layer_scatter_modes = kw["layer_scatter_modes"]
            self.post_attention_layernorm = kw["post_attention_layernorm"]
            self.is_last_layer = kw["is_last_layer"]

        should_use_reduce_scatter = lambda *a: False
        should_fuse_mlp_allreduce_with_next_layer = lambda *a: False

    ns.update(
        DeepseekV2AttentionMLA=Attention,
        Glm5NextLinearAttention=Attention,
        Glm5NextMoE=Moe,
        AscendMHCLayerCommunicator=Communicator,
        RMSNorm=lambda *a, **kw: torch.nn.Identity(),
    )
    ns["modules"]["sglang.srt.utils"] = NS(is_npu=lambda: True)
    load("models/glm5_next.py", ["Glm5NextDecoderLayer"], ns)
    decoder = ns["Glm5NextDecoderLayer"]
    decoder.hc_attn_pre = decoder.hc_ffn_pre = decoder.hc_post = lambda *a: None
    cfg = NS(
        hidden_size=6,
        intermediate_size=64,
        hidden_act="silu",
        swiglu_limit=10,
        rope_theta=1000,
        rope_scaling=None,
        max_position_embeddings=100,
        is_kda_layer=lambda _: False,
        rms_norm_eps=1e-5,
        mhc=True,
        hc_mult=2,
        num_hidden_layers=45,
        first_k_dense_replace=3,
        moe_layer_freq=1,
        n_routed_experts=288,
        q_lora_rank=1,
        num_attention_heads=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=0,
        v_head_dim=4,
        kv_lora_rank=2,
    )
    return ns, decoder, cfg, Communicator, forward


class DecoderTests(unittest.TestCase):
    def test_actual_constructor_only_target_first_three_and_no_wrapper(self):
        for rank in range(32):
            group = NS(world_size=16, rank_in_group=rank % 16)
            ns, decoder, cfg, _, _ = decoder_fixture(rank, group=group)
            for layer_id in range(3):
                layer = decoder(
                    cfg,
                    layer_id,
                    quant_config=ns["Quant"](),
                    prefix=f"model.layers.{layer_id}",
                )
                self.assertIs(layer._dense_tp_group, group)
                self.assertEqual(layer.mlp.down_proj.tp_rank, rank % 16)
                self.assertEqual(layer.mlp.down_proj.tp_size, 16)
                self.assertFalse(layer.mlp.down_proj.reduce_results)
                self.assertIn("mlp.gate_up_proj.weight", dict(layer.named_parameters()))
            for layer_id, draft in ((3, False), (44, False), (45, True)):
                layer = decoder(
                    cfg, layer_id, quant_config=ns["Quant"](), is_nextn=draft
                )
                self.assertIsNone(layer._dense_tp_group)
                self.assertIsInstance(layer.mlp, ns["Glm5NextMoE"])
        for dense in (None, 1, 32):
            ns, decoder, cfg, _, _ = decoder_fixture(0, dense=dense)
            layer = decoder(cfg, 0, quant_config=ns["Quant"]())
            self.assertIsNone(layer._dense_tp_group)
            self.assertEqual(layer.mlp.down_proj.tp_size, 1 if dense == 1 else 32)
        ns, decoder, cfg, _, _ = decoder_fixture(0)
        with self.assertRaises(RuntimeError):
            decoder(cfg, 0, quant_config=ns["Quant"]())
        ns, decoder, cfg, _, _ = decoder_fixture(0, tp=16)
        layer = decoder(cfg, 0, quant_config=ns["Quant"]())
        self.assertIsNone(layer._dense_tp_group)
        self.assertEqual(layer.mlp.down_proj.tp_size, 16)
        self.assertTrue(layer.mlp.down_proj.reduce_results)

    def test_actual_decoder_mhc_full_dp4_gather_16_sum_and_own_dp_scatter(self):
        for lengths in ((3, 1, 0, 2), (1, 0, 0, 0), (0, 0, 0, 3), (0, 0, 0, 0)):
            P, H = max(lengths), 6
            locals_ = []
            for dp, count in enumerate(lengths):
                x = torch.zeros(P if count else 0, H)
                if count:
                    x[:count] = (dp + 1) * 10 + torch.arange(count).view(-1, 1)
                locals_.append(x)
            full = torch.cat([x if x.shape[0] else torch.zeros(P, H) for x in locals_])
            partials = [full * (r % 16 + 1) for r in range(32)]
            reductions = []
            for rank in range(32):
                dp = rank // 8

                class Group:
                    world_size = 16
                    rank_in_group = rank % 16

                    def all_reduce(self, tensor):
                        self_rank = rank
                        self.assert_partial(tensor, partials[self_rank])
                        members = range(self_rank // 16 * 16, self_rank // 16 * 16 + 16)
                        reductions.append(tuple(members))
                        return sum(
                            (partials[r] for r in members), torch.zeros_like(tensor)
                        )

                    assert_partial = staticmethod(torch.testing.assert_close)

                group = Group()
                ns, decoder, cfg, communicator, forward = decoder_fixture(
                    rank, group=group
                )
                ns.update(
                    ScatterMode=NS(SCATTERED="SCATTERED"),
                    attention_tensor_model_parallel_all_reduce=lambda x: x,
                    get_attention_dp_rank=lambda: dp,
                    get_attn_tensor_model_parallel_rank=lambda: rank % 8,
                    get_tensor_model_parallel_world_size=lambda: 32,
                    get_global_dp_buffer=lambda _: torch.empty_like(full),
                    world_dp_gather_enabled=lambda: False,
                    _note_dp_gather_in_prefill_graph=lambda: None,
                    hc_contract=lambda x, _: x,
                )

                def memcpy(dst, src, a, start, count, reverse):
                    start, count = int(start), int(count)
                    if reverse:
                        dst[:count].copy_(src[start : start + count])
                    else:
                        dst[start : start + count].copy_(src[:count])

                def global_gather(x):
                    contribution = torch.zeros_like(full)
                    if rank % 8 == 0 and lengths[dp]:
                        contribution[dp * P : (dp + 1) * P] = locals_[dp]
                    self.assertTrue(torch.equal(x, contribution))
                    return full.clone()

                ns.update(memcpy=memcpy, tensor_model_parallel_all_reduce=global_gather)
                load(
                    "layers/dp_attention.py",
                    ["get_dp_local_info", "_dp_gather_via_all_reduce", "dp_scatter"],
                    ns,
                )
                ns["modules"]["sglang.srt.layers.dp_attention"] = NS(
                    _dp_gather_via_all_reduce=ns["_dp_gather_via_all_reduce"]
                )
                load(
                    "layers/communicator_mhc.py",
                    [
                        "AscendMHCLayerCommunicator.prepare_mlp",
                        "AscendMHCLayerCommunicator.postprocess_layer",
                    ],
                    ns,
                )
                communicator.prepare_mlp = ns["prepare_mlp"]
                communicator.postprocess_layer = ns["postprocess_layer"]
                communicator.prepare_attn = lambda self, x, res, fb: (x, res)
                layer = decoder(cfg, 0, quant_config=ns["Quant"]())
                comm = layer.layer_communicator
                comm._context = NS(
                    attn_dp_size=4, attn_tp_size=8, attn_tp_rank=rank % 8
                )
                comm.mhc = NS(
                    attn_to_mlp=lambda x, res, **kw: (x, res),
                    mlp_combine=lambda x, res: x,
                    reset_aux=lambda: None,
                    hc_mult=2,
                )
                original = layer.mlp

                class TaggedMLP(torch.nn.Module):
                    gate_up_proj = NS(gather_output=False)
                    down_proj = NS(
                        use_decode_attn_tp=False, use_dp_attention_reduce=False
                    )

                    def forward(self, x, *a):
                        self_called = rank
                        torch.testing.assert_close(x, full)
                        return x * (self_called % 16 + 1)

                layer.mlp = TaggedMLP()
                fb = NS(
                    global_num_tokens_gpu=torch.tensor([P] * 4), dp_local_start_pos=None
                )
                output, residual, _ = layer(
                    torch.zeros(1), locals_[dp], fb, torch.zeros_like(locals_[dp])
                )
                self.assertTrue(torch.equal(output, locals_[dp] * 136))
                self.assertIsNone(residual)
                if P:
                    forward.sp_active = True
                    with self.assertRaises(RuntimeError):
                        layer(
                            torch.zeros(1),
                            locals_[dp],
                            fb,
                            torch.zeros_like(locals_[dp]),
                        )
                del original
            self.assertEqual(len(reductions), 32 if P else 0)
            self.assertTrue(all(len(members) == 16 for members in reductions))
            if P:
                self.assertFalse(
                    torch.equal(sum(partials), full * 136),
                    "negative control: TP32 would sum two replicas",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
