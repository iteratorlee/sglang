"""Opt-in local-asset equivalence tests; no model weights, network, or GPU imports.

Set SGLANG_GLM53_CPU_TOKENIZER_DIR to the local assets. Frozen fixtures additionally
require SGLANG_GLM53_CPU_ALLOW_LARGE=1 and SGLANG_GLM53_CPU_FIXTURE_DIR, only after
benchmark-window clearance. Run in a standalone Python process with taskset -c 0,1.

To avoid importing the GPU serving stack, compile the actual protocol declarations,
selected serving methods, and content helpers from their source AST, with unchanged
function bodies. The real HF tokenizer executes the real Jinja asset. Ancillary
serving configuration uses no function-call parser; these tests are not a full
server/HTTP or grammar-parser integration test. Unsupported branches fail loudly.
"""

import ast
import copy
import gzip
import hashlib
import importlib.util
import json
import logging
import os
import sys
import time
import types
import unittest
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from unittest.mock import patch

ROOT = (
    Path(os.environ["SGLANG_GLM53_CPU_SOURCE_ROOT"])
    if "SGLANG_GLM53_CPU_SOURCE_ROOT" in os.environ
    else Path(__file__).resolve().parents[5]
)
PYTHON = ROOT / "python/sglang"
OPENAI = PYTHON / "srt/entrypoints/openai"
NS = types.SimpleNamespace


def load_helper():
    spec = importlib.util.spec_from_file_location(
        "glm53_real_cpu_helper",
        os.environ.get(
            "SGLANG_GLM53_CPU_HELPER_PATH", OPENAI / "glm53_chat_tokenization.py"
        ),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute_nodes(path, namespace, names):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    assert len(nodes) == len(names), (path, names)
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    ast.fix_missing_locations(future)
    exec(
        compile(ast.Module(body=[future, *nodes], type_ignores=[]), str(path), "exec"),
        namespace,
    )


def source_renderer(stack):
    import jinja2
    import orjson
    import pydantic
    import transformers.utils.chat_template_utils as hf_chat_utils

    protocol = types.ModuleType("_glm53_cpu_protocol")
    stack.enter_context(patch.dict(sys.modules, {protocol.__name__: protocol}))
    protocol.__dict__.update(json=json, BaseModel=pydantic.BaseModel)
    execute_nodes(
        PYTHON / "utils.py", protocol.__dict__, {"convert_json_schema_to_str"}
    )
    path = OPENAI / "protocol.py"
    tree = ast.parse(path.read_text())
    # The optional StructuralTag import can import Torch; it is only a type here.
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.ImportFrom) and node.module == "sglang.utils")
        and not (isinstance(node, ast.Try) and "xgrammar" in ast.unparse(node))
    ]
    from typing import Any

    protocol.StructuralTag = Any
    exec(compile(tree, str(path), "exec"), protocol.__dict__)

    module = types.ModuleType("sglang.srt.entrypoints.openai.serving_chat")
    stack.enter_context(patch.dict(sys.modules, {module.__name__: module}))
    ns = module.__dict__
    ns.update(vars(protocol))
    ns.update(
        __name__="sglang.srt.entrypoints.openai.serving_chat",
        copy=copy,
        json=json,
        orjson=orjson,
        Enum=Enum,
        logger=logging.getLogger(__name__),
        jinja2=jinja2,
        hf_chat_utils=hf_chat_utils,
        _CHAT_TEMPLATE_CLIENT_ERRORS=(jinja2.TemplateError, TypeError),
        _CHAT_TEMPLATE_CACHE_MAX_SIZE=128,
        MEDIA_URL_PART_TYPES=("image_url", "input_image", "video_url", "audio_url"),
        envs=NS(SGLANG_DEFAULT_THINKING=NS(get=lambda: False)),
    )
    execute_nodes(
        PYTHON / "srt/parser/hunyuan_reasoning.py",
        ns,
        {
            "uses_hunyuan_reasoning_effort",
            "normalize_hunyuan_reasoning_effort",
        },
    )
    execute_nodes(
        PYTHON / "srt/parser/jinja_template_utils.py",
        ns,
        {
            "process_content_for_template_format",
            "_is_var_access",
            "_is_attr_access",
            "_is_var_or_elems_access",
            "_try_extract_ast",
            "detect_jinja_template_content_format",
            "jinja_template_may_reorder_tool_results",
        },
    )
    encoding = {}
    execute_nodes(OPENAI / "chat_encoding.py", encoding, {"spec_renders_prompt_ids"})
    ns["chat_encoding"] = NS(**encoding)
    path = OPENAI / "serving_chat.py"
    execute_nodes(
        path,
        ns,
        {
            "ThinkingMode",
            "normalize_tool_content",
            "parse_tool_call_arguments",
            "normalize_assistant_tool_call_arguments",
        },
    )
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "OpenAIServingChat"
    )
    methods = set(load_helper()._RENDER_METHODS) | {
        "_normalize_reasoning_effort",
        "_effective_tools",
        "_patch_reasoning_skip_special_tokens",
        "_get_reasoning_from_request",
        "_canonicalize_tool_message_order",
        "_sort_tool_message_run",
    }
    cls.bases = []
    cls.body = [node for node in cls.body if getattr(node, "name", None) in methods]
    assert len(cls.body) == len(methods)
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    ast.fix_missing_locations(future)
    exec(
        compile(ast.Module(body=[future, cls], type_ignores=[]), str(path), "exec"), ns
    )
    return module, protocol


@unittest.skipUnless(
    os.environ.get("SGLANG_GLM53_CPU_TOKENIZER_DIR"), "local assets opt-in required"
)
class TestGlm53RealCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from contextlib import ExitStack

        cls.stack = ExitStack()
        cls.addClassCleanup(cls.stack.close)
        # Even USE_TORCH=0 does not prevent imports in Transformers 5.6.
        if any(
            sys.modules.get(name) is not None
            for name in ("torch", "torchvision", "torchaudio", "torch_npu")
        ):
            raise RuntimeError("run this CPU test in a fresh process without Torch")
        cls.stack.enter_context(
            patch.dict(
                sys.modules,
                {
                    "torch": None,
                    "torchvision": None,
                    "torchaudio": None,
                    "torch_npu": None,
                },
            )
        )
        cls.stack.enter_context(
            patch.dict(
                os.environ,
                {
                    "CUDA_VISIBLE_DEVICES": "",
                    "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
                    "USE_TORCH": "0",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                    "RAYON_NUM_THREADS": "2",
                    "OMP_NUM_THREADS": "2",
                    "OPENBLAS_NUM_THREADS": "2",
                },
            )
        )
        if hasattr(os, "sched_getaffinity"):
            allowed = sorted(os.sched_getaffinity(0))
            os.sched_setaffinity(0, allowed[:2])
        from transformers import TokenizersBackend

        cls.assets = Path(os.environ["SGLANG_GLM53_CPU_TOKENIZER_DIR"])
        cls.tokenizer = TokenizersBackend.from_pretrained(
            cls.assets, local_files_only=True
        )
        cls.helper = load_helper()
        cls.stack.enter_context(patch.dict(os.environ, {cls.helper.OPT_IN_ENV: "1"}))
        cls.module, cls.protocol = source_renderer(cls.stack)

    def make_serving(self, model="default"):
        serving = self.module.OpenAIServingChat()
        serving.tokenizer_manager = NS(
            tokenizer=self.tokenizer,
            served_model_name=model,
            model_path=str(self.assets),
            config_value=lambda _: "cpu-test",
            model_config=NS(
                is_multimodal=True,
                hf_config=NS(
                    model_type="glm5_next",
                    architectures=["Glm5NextForConditionalGeneration"],
                ),
            ),
        )
        serving.template_manager = NS(
            chat_template_name=None,
            reasoning_config=None,
            jinja_template_content_format=self.module.detect_jinja_template_content_format(
                self.tokenizer.chat_template
            ),
            jinja_template_may_reorder_tool_results=self.module.jinja_template_may_reorder_tool_results(
                self.tokenizer.chat_template
            ),
        )
        serving.chat_encoding_spec = None
        serving.is_gpt_oss = serving.is_gemma4 = False
        serving.reasoning_parser = "glm45"
        serving._reasoning_detector = NS(reasoning_default="always")
        serving.tool_call_parser = None
        serving.default_chat_template_kwargs = {}
        serving._tokenizer_auto_adds_specials = self.tokenizer.encode(
            "x"
        ) != self.tokenizer.encode("x", add_special_tokens=False)
        serving._chat_template_cache = OrderedDict()
        return serving

    def compare(self, data, label, expect_reuse=True):
        # Actual request schemas and methods, without a hand-written renderer.
        request = self.protocol.ChatCompletionRequest.model_validate(
            copy.deepcopy(data)
        )
        serving = self.make_serving(request.model)
        serving._normalize_reasoning_effort(request)
        start = time.perf_counter()
        guard = self.helper.Glm53ChatTokenizationGuard(serving, request, True)
        setup_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        processed = serving._process_messages(request, True)
        render_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        candidate = guard.try_reuse(processed)
        proof_ms = (time.perf_counter() - start) * 1000
        kind, prompt = serving._engine_prompt(processed, True)
        self.assertEqual(kind, "text")
        start = time.perf_counter()
        reference = self.tokenizer([prompt])["input_ids"][0]
        encode_ms = (time.perf_counter() - start) * 1000
        if expect_reuse:
            self.assertIsNotNone(candidate, label)
        else:
            self.assertIsNone(candidate, label)
        if candidate is not None:
            self.assertEqual(candidate, reference, label)
        report = {
            "case": label,
            "tokens": len(reference),
            "reused": candidate is not None,
            "ids_equal": candidate == reference if candidate is not None else None,
            "render_ms": round(render_ms, 3),
            "final_encode_ms": round(encode_ms, 3),
            "guard_ms": round(setup_ms + proof_ms, 3),
            "ids_sha256": hashlib.sha256(
                json.dumps(reference, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        print(json.dumps(report), flush=True)
        self.assertIsNone(sys.modules.get("torch"))
        return processed, report

    def test_small_actual_renderer_cases(self):
        texts = [
            "hello",
            "  a\t b\r\n\n尾部  ",
            "é e\u0301 中文 👩🏽‍💻 \u0000 \ufffd",
            "hello . I ' m testing !",
            "<|user|>[gMASK]<sop><think></think><|endoftext|>",
        ]
        for index, text in enumerate(texts):
            with self.subTest(index=index):
                self.compare(
                    {"messages": [{"role": "user", "content": text}]}, f"text-{index}"
                )
        self.compare(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "a"},
                            {"type": "text", "text": "b"},
                        ],
                    }
                ]
            },
            "text-parts",
        )
        messages = [
            {"role": "user", "content": "old question"},
            {
                "role": "assistant",
                "content": "old answer",
                "reasoning_content": "UNIQUE_OLD_THINKING",
            },
            {"role": "user", "content": "new question"},
        ]
        kept, _ = self.compare(
            {"messages": messages, "chat_template_kwargs": {"clear_thinking": False}},
            "thinking-kept",
        )
        cleared, _ = self.compare(
            {"messages": messages, "chat_template_kwargs": {"clear_thinking": True}},
            "thinking-cleared",
        )
        self.assertIn("UNIQUE_OLD_THINKING", kept.prompt)
        self.assertNotIn("UNIQUE_OLD_THINKING", cleared.prompt)
        self.assertNotEqual(kept.prompt_ids, cleared.prompt_ids)
        self.compare(
            {
                "messages": [
                    {"role": "user", "content": "continue"},
                    {"role": "assistant", "content": "prefix"},
                ],
                "continue_final_message": True,
            },
            "continuation-fallback",
            expect_reuse=False,
        )

    def test_advisory_endpoint_integration_and_fallback(self):
        # Execute the complete advisory method with real protocol/template/IDs.
        # The HTTP/grammar/GPU stack remains outside this CPU-only test.
        spec = importlib.util.spec_from_file_location(
            "glm53_real_cpu_affinity", OPENAI / "chat_affinity.py"
        )
        affinity = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(affinity)
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.entrypoints.openai.protocol": self.protocol,
                "sglang.srt.entrypoints.openai.glm53_chat_tokenization": self.helper,
            },
        ):
            serving = self.make_serving()
            serving._validate_request = lambda request: None
            service = affinity.ChatAffinityTokenizer(serving)
            self.addCleanup(service.close)
            data = {
                "model": "default",
                "messages": [{"role": "user", "content": "中文 café\n<|user|>"}],
            }
            with patch.dict(os.environ, {self.helper.OPT_IN_ENV: "0"}):
                reference = service._tokenize(data)
            self.assertTrue(reference["supported"])
            self.assertEqual(reference["tokenization_path"], "reference")
            with patch.dict(os.environ, {self.helper.OPT_IN_ENV: "1"}):
                actual = service._tokenize(data)
            self.assertTrue(actual["supported"])
            self.assertEqual(actual["tokenization_path"], "verified_render_ids")
            self.assertEqual(actual["tokens"], reference["tokens"])
            explicit = service._tokenize(dict(data, input_ids=reference["tokens"]))
            self.assertTrue(explicit["supported"])
            self.assertEqual(explicit["tokenization_path"], "request_input_ids")
            self.assertEqual(explicit["tokens"], reference["tokens"])
            media = copy.deepcopy(data)
            media["messages"][0]["content"] = [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AA=="},
                }
            ]
            self.assertFalse(service._tokenize(media)["supported"])

    def test_real_profile_and_unsafe_mutations(self):
        from tokenizers import (
            AddedToken,
            decoders,
            normalizers,
            pre_tokenizers,
            processors,
        )

        t = self.tokenizer
        b = t.backend_tokenizer
        self.assertIsNotNone(self.helper._tokenizer_profile(t))
        self.assertEqual(b.num_special_tokens_to_add(False), 0)
        for text in [
            "",
            "ordinary",
            *[item.content for item in b.get_added_tokens_decoder().values()],
        ]:
            with self.subTest(text=text):
                self.assertEqual(
                    t.encode(text, add_special_tokens=False), t([text])["input_ids"][0]
                )
                self.assertEqual(t.decode(t.encode(text)), text)
        mutations = [
            (t, "clean_up_tokenization_spaces", True),
            (t, "split_special_tokens", True),
            (b, "normalizer", normalizers.NFC()),
            (b.model, "dropout", 0.1),
            (b, "pre_tokenizer", pre_tokenizers.ByteLevel(add_prefix_space=True)),
            (b, "decoder", decoders.WordPiece()),
            (
                b,
                "post_processor",
                processors.TemplateProcessing(
                    single="<bos> $A", special_tokens=[("<bos>", 0)]
                ),
            ),
        ]
        for obj, name, value in mutations:
            with self.subTest(name=name):
                old = getattr(obj, name)
                try:
                    setattr(obj, name, value)
                    self.assertIsNone(self.helper._tokenizer_profile(t))
                finally:
                    setattr(obj, name, old)
        with patch.object(t, "encode", lambda _: [0]):
            self.assertIsNone(self.helper._tokenizer_profile(t))
        with patch.dict(sys.modules, {"transformers": NS(__version__="unknown")}):
            self.assertIsNone(self.helper._tokenizer_profile(t))
        original_added = next(
            item for item in b.get_added_tokens_decoder().values() if item.special
        )
        b.add_special_tokens(
            [AddedToken(original_added.content, special=True, lstrip=True)]
        )
        try:
            self.assertIsNone(self.helper._tokenizer_profile(t))
        finally:
            b.add_special_tokens([original_added])
        self.assertIsNotNone(self.helper._tokenizer_profile(t))
        b.enable_truncation(16)
        try:
            self.assertIsNone(self.helper._tokenizer_profile(t))
        finally:
            b.no_truncation()
        b.enable_padding(length=16)
        try:
            self.assertIsNone(self.helper._tokenizer_profile(t))
        finally:
            b.no_padding()
        # Demonstrate why arbitrary ID roundtrip identity is not a premise.
        mismatches = []
        for i in range(256):
            if t.encode(t.decode([i])) != [i]:
                mismatches.append(i)
        self.assertTrue(mismatches)
        print(
            json.dumps({"non_identity_single_id_examples": mismatches[:8]}), flush=True
        )

    @unittest.skipUnless(
        os.environ.get("SGLANG_GLM53_CPU_ALLOW_LARGE") == "1",
        "main benchmark-window clearance required",
    )
    def test_frozen_requests(self):
        directory = Path(os.environ["SGLANG_GLM53_CPU_FIXTURE_DIR"])
        paths = sorted(directory.glob("*.request.json.gz"))
        self.assertEqual(len(paths), 44)
        reports = []
        for path in paths:
            with self.subTest(path=path.name):
                data = json.loads(gzip.decompress(path.read_bytes()))
                _, report = self.compare(data, path.name)
                reports.append(report)
        print(
            json.dumps(
                {
                    "frozen_summary": {
                        "cases": len(reports),
                        "equal": sum(r["ids_equal"] is True for r in reports),
                        "tokens_min": min(r["tokens"] for r in reports),
                        "tokens_max": max(r["tokens"] for r in reports),
                        "render_ms_sum": round(sum(r["render_ms"] for r in reports), 3),
                        "final_encode_ms_sum": round(
                            sum(r["final_encode_ms"] for r in reports), 3
                        ),
                        "guard_ms_sum": round(sum(r["guard_ms"] for r in reports), 3),
                    }
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
