"""Worker graph gate regression. Run with the CPU-only contract suite's env.

Build the real ServerArgs record from all actual field namespaces and its
actual assembly code. Execute graph parsing/declarations/views and bootstrap
source unchanged; only unrelated imports and device/transport are boundaries.
"""

from contextlib import contextmanager
import dataclasses
import functools
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest import mock

import msgspec

import test_glm53_dense_tp16_cpu as contracts


@contextmanager
def real_config_sources():
    modules = {}
    installed = {}

    def source(path, names=None):
        name = "_dense16_graph_cpu_" + path.replace("/", "_").replace(".", "_")
        module = ModuleType(name)
        ns = module.__dict__
        ns.update(contracts.namespace())
        ns["__name__"] = name
        ns["modules"].update(modules)
        installed[name] = module
        sys.modules[name] = module
        if names is None:
            exec(
                compile((contracts.SRT / path).read_text(), str(contracts.SRT / path), "exec"),
                ns,
            )
        else:
            contracts.load(path, names, ns)
        return module

    # These accessors are not invoked by graph parsing. No platform probe runs.
    modules["sglang.srt.runtime_context"] = NS(get_exec=lambda: None)
    modules["sglang.srt.utils"] = NS(is_cuda=lambda: False)
    try:
        for path in ("arg_utils", "choices", "field_order", "argparse_actions"):
            modules[f"sglang.srt.arg_groups.{path}"] = source(f"arg_groups/{path}.py")
        graph = source("model_executor/cuda_graph_config.py")
        modules["sglang.srt.model_executor.cuda_graph_config"] = graph
        # Parser metadata unrelated to this test; no LoRA or model loading runs.
        modules["sglang.srt.lora.lora_registry"] = NS(LoRARef=str)
        common = source(
            "utils/common.py", ["human_readable_int", "json_list_type", "nullable_str"]
        )
        common.LORA_TARGET_ALL_MODULES = "all"
        common.SUPPORTED_LORA_TARGET_MODULES = []
        modules["sglang.srt.utils.common"] = common
        fields = source("arg_groups/fields/__init__.py")
        ns = contracts.namespace()
        for path in (contracts.SRT / "arg_groups/fields").glob("*.py"):
            if path.name != "__init__.py":
                module = source(f"arg_groups/fields/{path.name}")
                ns.update({k: v for k, v in vars(module).items() if isinstance(v, type)})
        ns.update(
            msgspec=msgspec,
            functools=functools,
            collect_input_fields=fields.collect_input_fields,
        )
        contracts.load("server_args.py", ["ServerArgs", "_underscore_field_names"], ns)
        text = (contracts.SRT / "server_args.py").read_text()
        assembly = "_INPUT_NAMESPACES = [" + text.split("_INPUT_NAMESPACES = [", 1)[1]
        exec(
            compile(
                assembly.split("# Module-level ServerArgs helpers", 1)[0],
                str(contracts.SRT / "server_args.py"),
                "exec",
            ),
            ns,
        )
        arg_utils = modules["sglang.srt.arg_groups.arg_utils"]
        ns.update(record_fields=arg_utils.record_fields)
        contracts.load(
            "arg_groups/model_override_base.py",
            ["ResolvedView", "ResolvingConfig", "resolving_view"],
            ns,
        )
        ns.update(
            is_record=arg_utils.is_record,
            field_names=arg_utils.field_names,
            get_context=lambda: NS(server_args=None),
        )
        contracts.load("arg_groups/overrides.py", ["declare_resolution"], ns)
        ns.update(vars(graph))
        contracts.load("arg_groups/cuda_graph_hook.py", ["parse_cuda_graph_config"], ns)
        yield ns
    finally:
        for name in installed:
            sys.modules.pop(name, None)


class WorkerGraphTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(mock.patch.dict(contracts.os.environ, contracts.ENV))
        self.config = self.enterContext(real_config_sources())
        self.helper = contracts.helper_ns()
        # Use the canonical view's actual implementation, including stash reads.
        self.helper["modules"]["sglang.srt.arg_groups.model_override_base"] = NS(
            resolving_view=self.config["resolving_view"]
        )

    def args(self, **changes):
        values = vars(contracts.profile(cuda_graph_config=None))
        values.update(model_path="offline-model", served_model_name="glm", **changes)
        args = self.config["ServerArgs"](**values)
        self.assertIsInstance(args, msgspec.Struct)
        self.config["parse_cuda_graph_config"](args)
        return args

    def test_raw_worker_record_and_cli_resolving_view_agree(self):
        args = self.args()
        view = self.config["resolving_view"](args)
        self.assertIsNone(args.cuda_graph_config)
        self.assertIsInstance(view.cuda_graph_config, self.config["CudaGraphConfig"])
        self.assertEqual(
            dataclasses.asdict(view.cuda_graph_config)["prefill"]["backend"], "disabled"
        )
        gate = self.helper["dense_tp16_enabled"]
        self.assertTrue(gate(view, is_npu=True))
        self.assertTrue(gate(args, is_npu=True))
        self.assertIsNone(args.cuda_graph_config, "gate must not mutate raw input")

    def test_explicit_phase_graph_still_rejected_despite_legacy_disable(self):
        for phase in ("prefill", "decode"):
            for backend in ("full", "breakable", "tc_piecewise"):
                for mode in ("json", "convenience", "typed"):
                    with self.subTest(phase=phase, backend=backend, mode=mode):
                        changes = {f"cuda_graph_backend_{phase}": backend}
                        if mode == "json":
                            changes = {
                                "cuda_graph_config": {phase: {"backend": backend}}
                            }
                        elif mode == "typed":
                            graph = self.config["default_cuda_graph_config"]()
                            # Typed input serializes only differences from
                            # defaults; an unchanged backend is not explicit.
                            if getattr(graph, phase).backend == backend:
                                continue
                            getattr(graph, phase).backend = backend
                            changes = {"cuda_graph_config": graph}
                        args = self.args(**changes)
                        self.assertTrue(args.disable_cuda_graph)
                        for cfg in (args, self.config["resolving_view"](args)):
                            with self.assertRaisesRegex(
                                ValueError, "both phases must be disabled"
                            ):
                                self.helper["dense_tp16_enabled"](cfg, is_npu=True)

    def test_real_cli_then_worker_bootstrap_reaches_transport_boundary(self):
        args = self.args()
        ns = self.helper

        class AtBoundary(Exception):
            pass

        def stop(*a, **kw):
            raise AtBoundary

        ns.update(
            resolving_view=self.config["resolving_view"],
            get_platform=lambda: NS(is_npu=True),
            is_npu=lambda: True,
            is_runai_obj_uri=lambda _: False,
            get_parallel=lambda: self.config["resolving_view"](args),
            get_exec=lambda: NS(
                moe=NS(is_ep_joiner=False, is_ep_scale_joiner=False,
                       moe_a2a_backend="deepep")
            ),
            init_distributed_environment=stop,
        )
        ns["modules"].update(
            {
                contracts.HELPER: NS(
                    dense_tp16_enabled=ns["dense_tp16_enabled"],
                    validate_dense_tp16_model=ns["validate_dense_tp16_model"],
                ),
                "sglang.srt.arg_groups.lora_hook": NS(check_lora_server_args=stop),
            }
        )
        contracts.load("arg_groups/validation_hook.py", ["check_server_args"], ns)
        contracts.load("distributed/bootstrap.py", ["_init_parallel_groups"], ns)
        # CLI uses a resolving view; the worker passes the original record.
        with self.assertRaises(AtBoundary):
            ns["check_server_args"](args)
        text = NS(
            hidden_size=4096, intermediate_size=12288,
            first_k_dense_replace=3, moe_layer_freq=1,
            mhc=True, hidden_act="silu", swiglu_limit=10,
        )
        model = NS(
            hf_config=NS(architectures=["Glm5NextForConditionalGeneration"]),
            hf_text_config=text,
        )
        with self.assertRaises(AtBoundary):
            ns["_init_parallel_groups"](
                backend="hccl", dist_init_method="offline", server_args=args,
                model_config=model, gpu_id=0, tp_rank=0, tp_size=32,
                pp_rank=0, pp_size=1, attn_dp_size=4, attn_cp_size=1,
                moe_ep_size=32, moe_dp_size=1, dcp_size=1,
            )
        self.assertIsNone(args.cuda_graph_config)

    def test_unresolved_and_missing_legacy_disable_fail_closed(self):
        args = self.config["ServerArgs"](
            **vars(contracts.profile(cuda_graph_config=None)), model_path="offline-model"
        )
        with self.assertRaisesRegex(ValueError, "both phases must be disabled"):
            self.helper["dense_tp16_enabled"](args, is_npu=True)
        args = self.args(
            disable_cuda_graph=False,
            disable_prefill_cuda_graph=True,
            disable_decode_cuda_graph=True,
        )
        with self.assertRaisesRegex(ValueError, "disable_cuda_graph"):
            self.helper["dense_tp16_enabled"](args, is_npu=True)


if __name__ == "__main__":
    unittest.main()
