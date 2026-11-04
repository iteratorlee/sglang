"""CPU-only protocol/route tests using real serving methods and a tiny HF tokenizer.

No model downloads or GPU imports. Reuse the existing source-AST harness to load
unchanged Chat rendering/validation/conversion bodies, then import the complete
Anthropic and affinity modules. Scheduler IO is a recording SimpleNamespace;
this does not test actual generation, vision token expansion, or PD transport.

Run (the outer timeout also bounds imports, native code and interpreter exit):
    timeout --signal=TERM --kill-after=5s 60s python -m unittest discover \
        -s test/registered/unit/entrypoints/anthropic -p test_pd_tokenize_cpu.py -v

Every async test has a 10-second deadline and a test-only selector wake-up timer.
Restricted sandboxes can deny asyncio's cross-thread socket send with EPERM;
the timer lets completed worker futures be consumed without changing the worker
or masking an executor deadlock. A deadline failure dumps all thread stacks.
"""

import ast
import asyncio
import faulthandler
import functools
import importlib.util
import json
import logging
import os
import sys
import threading
import time
import types
import unittest
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[5]
PYTHON = ROOT / "python"
ENTRY = PYTHON / "sglang/srt/entrypoints"


async def run_bounded_cpu_test(coroutine, timeout=10):
    loop = asyncio.get_running_loop()
    wakeup = None

    def tick():
        nonlocal wakeup
        wakeup = loop.call_later(0.01, tick)

    tick()
    try:
        async with asyncio.timeout(timeout):
            return await coroutine
    except TimeoutError:
        faulthandler.dump_traceback()
        raise
    finally:
        wakeup.cancel()


def bounded_cpu_test(test):
    @functools.wraps(test)
    async def wrapped(self):
        return await run_bounded_cpu_test(test(self))

    return wrapped


def load_module(name, path, stack):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    stack.enter_context(patch.dict(sys.modules, {name: module}))
    spec.loader.exec_module(module)
    return module


def add_methods(path, target, names):
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == target.OpenAIServingChat.__name__
    )
    cls.bases = []
    cls.body = [n for n in cls.body if getattr(n, "name", None) in names]
    assert len(cls.body) == len(names)
    cls.name = "ExtraMethods"
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    tree = ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[]))
    exec(compile(tree, str(path), "exec"), vars(target))
    for name in names:
        setattr(target.OpenAIServingChat, name, vars(target.ExtraMethods)[name])


def cpu_modules(stack):
    stack.enter_context(
        patch.dict(
            os.environ,
            {
                "USE_TORCH": "0",
                "CUDA_VISIBLE_DEVICES": "",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "RAYON_NUM_THREADS": "1",
                "SGLANG_CHAT_AFFINITY_CPU_WORKERS": "1",
                "SGLANG_GLM53_REUSE_RENDER_IDS": "0",
            },
        )
    )
    stack.enter_context(
        patch.dict(
            sys.modules,
            {
                name: None
                for name in ("torch", "torchvision", "torchaudio", "torch_npu")
            },
        )
    )
    # Avoid package __init__ files that import the inference stack.
    packages = {}
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.entrypoints",
        "sglang.srt.entrypoints.openai",
        "sglang.srt.entrypoints.anthropic",
        "sglang.srt.parser",
        "sglang.srt.observability",
    ):
        module = types.ModuleType(name)
        module.__path__ = [str(PYTHON / name.replace(".", "/"))]
        packages[name] = module
    stack.enter_context(patch.dict(sys.modules, packages))
    harness = load_module(
        "_anthropic_chat_cpu_harness",
        ROOT
        / "test/registered/unit/entrypoints/openai/test_glm53_chat_tokenization_real_cpu.py",
        stack,
    )
    chat, protocol = harness.source_renderer(stack)
    stack.enter_context(
        patch.dict(sys.modules, {"sglang.srt.entrypoints.openai.protocol": protocol})
    )
    from jsonschema import Draft202012Validator, SchemaError

    ns = vars(chat)
    ns.update(
        Draft202012Validator=Draft202012Validator,
        SchemaError=SchemaError,
        _MEDIA_CONTENT_PART_TYPES={"image_url", "video_url", "audio_url"},
        get_model=lambda: NS(context_length=100000),
        get_serving=lambda: NS(allow_auto_truncate=False, return_input_ids=False),
        GenerateReqInput=lambda **kw: NS(**kw),
        ImageData=lambda **kw: NS(**kw),
        set_request_reasoning_end_token_ids=lambda *args: None,
    )
    ns["envs"].SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES = NS(get=lambda: False)
    harness.execute_nodes(
        ENTRY / "openai/chat_encoding.py", ns, {"spec_owns_reasoning_history"}
    )
    ns["chat_encoding"].spec_owns_reasoning_history = ns["spec_owns_reasoning_history"]
    utils = load_module(
        "sglang.srt.function_call.utils",
        PYTHON / "sglang/srt/function_call/utils.py",
        stack,
    )
    ns["normalize_json_schema_types"] = utils.normalize_json_schema_types
    harness.execute_nodes(
        ENTRY / "openai/serving_chat.py",
        ns,
        {
            "_extract_max_dynamic_patch",
            "_build_video_config",
            "_extract_video_question",
        },
    )
    harness.execute_nodes(
        PYTHON / "sglang/srt/parser/reasoning_parser.py",
        ns,
        {"StreamingParseResult", "BaseReasoningFormatDetector", "Glm45Detector"},
    )
    add_methods(
        ENTRY / "openai/serving_chat.py",
        chat,
        {
            "_validate_request",
            "_validate_media_content",
            "_convert_to_internal_request",
            "supports_native_reasoning_history",
            "wrap_reasoning_history",
            "apply_reasoning_enabled",
            "_reasoning_default_mode",
            "_get_reasoning_toggle_param",
            "_should_return_input_ids",
        },
    )
    # These base-class methods only inspect headers; no request reaches a scheduler.
    base = types.ModuleType("_anthropic_cpu_base")
    base.OpenAIServingChat = chat.OpenAIServingChat
    base.__dict__.update(
        Optional=ns["Optional"],
        List=ns["List"],
        Union=ns["Union"],
        Tuple=ns["Tuple"],
        logger=logging.getLogger(__name__),
    )
    # Compile the base helpers into a temporary class with the same name.
    tree = ast.parse((ENTRY / "openai/serving_base.py").read_text())
    base_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    base_cls.name = "OpenAIServingChat"
    base_cls.bases = []
    names = {
        "_resolve_lora_path",
        "_parse_model_parameter",
        "extract_custom_labels",
        "extract_routed_dp_rank_from_header",
        "extract_routing_key",
    }
    base_cls.body = [n for n in base_cls.body if getattr(n, "name", None) in names]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, base_cls], type_ignores=[])
            ),
            "cpu_base",
            "exec",
        ),
        vars(base),
    )
    for name in names:
        setattr(chat.OpenAIServingChat, name, vars(base.OpenAIServingChat)[name])
    # Load the real template detector without importing unrelated server config.
    overrides = NS(declare_resolution=None, resolving_view=None)
    stack.enter_context(
        patch.dict(
            sys.modules,
            {
                "sglang.srt.arg_groups.overrides": overrides,
                "sglang.srt.observability.req_time_stats": NS(
                    monotonic_time=time.perf_counter
                ),
            },
        )
    )
    load_module(
        "sglang.srt.parser.template_detection",
        PYTHON / "sglang/srt/parser/template_detection.py",
        stack,
    )
    anthropic = load_module(
        "sglang.srt.entrypoints.anthropic.protocol",
        ENTRY / "anthropic/protocol.py",
        stack,
    )
    affinity = load_module(
        "sglang.srt.entrypoints.openai.chat_affinity",
        ENTRY / "openai/chat_affinity.py",
        stack,
    )
    serving = load_module(
        "sglang.srt.entrypoints.anthropic.serving",
        ENTRY / "anthropic/serving.py",
        stack,
    )
    return chat, protocol, anthropic, affinity, serving


TEMPLATE = """{%- if tools %}TOOLS={{ tools | tojson }}\n{% endif -%}
SYSTEM-FIRST\n
{%- for message in messages %}{{ message.role }}={{ message.content | tojson }}
{%- if message.tool_calls %}CALLS={{ message.tool_calls | tojson }}{% endif %}\n
{%- endfor %}THINK={{ thinking | default(false) }};EFFORT={{ reasoning_effort | default('default') }};ASSISTANT="""

# The GLM-5.3 contract: effort header, native tools, unconditional thinking
# opener. It deliberately has no enable_thinking template control.
GLM53_TEMPLATE = """[gMASK]<sop>
{%- set effort = reasoning_effort if reasoning_effort is defined and reasoning_effort in ['low', 'high'] else 'max' -%}
<|system|>Reasoning Effort: {{ effort | capitalize }}
{%- for m in messages %}<|{{ m.role }}|>{{ m.content | tojson }}{% endfor -%}
{%- for tool in tools or [] %}<tool_call>{{ tool.function.name }}<arg_key>schema</arg_key><arg_value>{{ tool.function.parameters | tojson }}</arg_value></tool_call>{% endfor -%}
{%- if add_generation_prompt %}<|assistant|>{{ '<think>' }}{% endif -%}"""

THINKING_MODES = (
    ({"type": "disabled"}, False),
    ({"type": "adaptive"}, True),
    ({"type": "enabled", "budget_tokens": 1024}, True),
    ({"type": "adaptive", "display": "omitted"}, False),
    ({"type": "enabled", "budget_tokens": 1024, "display": "omitted"}, False),
)


class TestAnthropicPDCPU(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.stack = ExitStack()
        cls.addClassCleanup(cls.stack.close)
        (
            cls.chat_module,
            cls.chat_protocol,
            cls.protocol,
            cls.affinity,
            cls.serving_module,
        ) = cpu_modules(cls.stack)
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers
        from transformers import TokenizersBackend

        vocab = {
            char: i
            for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))
        }
        tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        cls.tokenizer = TokenizersBackend(tokenizer_object=tok, chat_template=TEMPLATE)

    def setUp(self):
        chat = self.chat_module.OpenAIServingChat()
        chat.tokenizer_manager = NS(
            tokenizer=self.tokenizer,
            served_model_name="cpu-model",
            model_path="cpu-fixture",
            config_value=lambda _: "v1",
            model_config=NS(is_multimodal=True, hf_config=NS(architectures=[])),
            create_abort_task=lambda _: None,
        )
        chat.template_manager = NS(
            chat_template_name=None,
            reasoning_config=None,
            jinja_template_content_format="string",
            jinja_template_may_reorder_tool_results=False,
        )
        chat.chat_encoding_spec = None
        chat.is_gpt_oss = chat.is_gemma4 = False
        chat.reasoning_parser = "glm45"
        chat._reasoning_detector = NS(
            reasoning_default="thinking",
            think_start_token="<think>",
            think_start_self_label="",
            think_end_token="</think>",
        )
        chat.tool_call_parser = None
        chat.default_chat_template_kwargs = {}
        chat.default_sampling_params = {}
        chat.allowed_custom_labels = None
        chat._tokenizer_auto_adds_specials = False
        chat._chat_template_cache = OrderedDict()
        self.chat = chat
        self.worker = self.affinity.ChatAffinityTokenizer(chat)
        self.addCleanup(self.worker.close)
        self.serving = self.serving_module.AnthropicServing(chat, self.worker)

    def request(self, **overrides):
        data = {
            "model": "cpu-model",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hello 世界"}],
        }
        data.update(overrides)
        return self.protocol.AnthropicMessagesRequest.model_validate(data)

    def stream(self, chunks, consumed):
        async def upstream(*args):
            for chunk in chunks:
                consumed.append(chunk)
                yield chunk

        self.enterContext(
            patch.object(self.chat, "_generate_chat_stream", upstream, create=True)
        )
        return self.serving._generate_anthropic_stream(
            None, None, self.request(stream=True, max_tokens=1), None
        )

    def use_glm53_template(self, template=GLM53_TEMPLATE):
        self.enterContext(patch.object(self.tokenizer, "chat_template", template))
        self.chat._reasoning_detector = self.chat_module.Glm45Detector()

    @bounded_cpu_test
    async def test_glm53_thinking_visibility_preserves_prompt_ids_and_counts(self):
        self.use_glm53_template()
        expected = None
        for thinking, _ in THINKING_MODES:
            with self.subTest(thinking=thinking):
                request = self.request(thinking=thinking)
                chat = self.serving._convert_to_chat_completion_request(request)
                self.assertTrue(self.chat._get_reasoning_from_request(chat))
                processed = self.chat._process_messages(chat, True)
                self.assertTrue(processed.prompt.endswith("<|assistant|><think>"))
                # The formerly supplied False toggle must render the same IDs.
                old = chat.model_copy(deep=True)
                self.chat.apply_reasoning_enabled(old, False)
                self.assertEqual(
                    processed.prompt_ids,
                    self.chat._process_messages(old, True).prompt_ids,
                )
                result = json.loads(
                    (await self.serving.handle_tokenize(request, None)).body
                )
                self.assertTrue(result["supported"], result)
                if expected is None:
                    expected = result["tokens"]
                self.assertEqual(result["tokens"], expected)
                count = await self.serving.handle_count_tokens(
                    self.protocol.AnthropicCountTokensRequest.model_validate(
                        request.model_dump()
                    ),
                    None,
                )
                self.assertEqual(json.loads(count.body)["input_tokens"], len(expected))

    @bounded_cpu_test
    async def test_glm53_nonstream_hides_reasoning_without_losing_answer_or_usage(self):
        self.use_glm53_template()
        for thinking, visible in THINKING_MODES:
            for truncated in (False, True):
                with self.subTest(thinking=thinking, truncated=truncated):
                    request = self.request(thinking=thinking)

                    async def generate(adapted, processed, raw_request):
                        self.assertTrue(adapted.require_reasoning)
                        detector = self.chat_module.Glm45Detector(
                            force_reasoning=self.chat._get_reasoning_from_request(
                                processed
                            )
                        )
                        parsed = detector.detect_and_parse(
                            "private rationale"
                            if truncated
                            else "private rationale</think>ANSWER"
                        )
                        return self.chat_protocol.ChatCompletionResponse(
                            id="cpu",
                            created=0,
                            model="cpu-model",
                            choices=[
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": parsed.normal_text,
                                        "reasoning_content": parsed.reasoning_text,
                                    },
                                    "finish_reason": "length" if truncated else "stop",
                                }
                            ],
                            usage={
                                "prompt_tokens": 37,
                                "completion_tokens": 1,
                                "total_tokens": 38,
                                "prompt_tokens_details": {"cached_tokens": 29},
                            },
                        )

                    with patch.object(
                        self.chat,
                        "_handle_non_streaming_request",
                        generate,
                        create=True,
                    ):
                        response = await self.serving.handle_messages(request, None)
                    self.assertEqual(response.status_code, 200, response.body)
                    body = json.loads(response.body)
                    self.assertEqual(
                        "".join(
                            block["text"]
                            for block in body["content"]
                            if block["type"] == "text"
                        ),
                        "" if truncated else "ANSWER",
                    )
                    self.assertEqual(
                        [
                            block["thinking"]
                            for block in body["content"]
                            if block["type"] == "thinking"
                        ],
                        ["private rationale"] if visible else [],
                    )
                    self.assertNotIn("</think>", response.body.decode())
                    self.assertEqual(
                        body["usage"],
                        {
                            "input_tokens": 8,
                            "cache_read_input_tokens": 29,
                            "output_tokens": 1,
                        },
                    )

    @bounded_cpu_test
    async def test_glm53_stream_visibility_and_hidden_only_usage(self):
        self.use_glm53_template()
        for thinking, visible in THINKING_MODES:
            for truncated in (False, True):
                with self.subTest(thinking=thinking, truncated=truncated):
                    request = self.request(thinking=thinking, stream=True, max_tokens=1)
                    chat = self.serving._convert_to_chat_completion_request(request)
                    count = json.loads(
                        (
                            await self.serving.handle_count_tokens(
                                self.protocol.AnthropicCountTokensRequest.model_validate(
                                    request.model_dump()
                                ),
                                None,
                            )
                        ).body
                    )["input_tokens"]

                    async def upstream(*args):
                        detector = self.chat_module.Glm45Detector(
                            force_reasoning=self.chat._get_reasoning_from_request(chat)
                        )
                        pieces = (
                            ["pri", "vate rationale"]
                            if truncated
                            else ["pri", "vate rationale</thi", "nk>A", "NSWER"]
                        )
                        for piece in pieces:
                            parsed = detector.parse_streaming_increment(piece)
                            yield self.stream_chunk(
                                {
                                    "content": parsed.normal_text,
                                    "reasoning_content": parsed.reasoning_text,
                                }
                            )
                        yield self.stream_chunk(
                            {}, finish_reason="length" if truncated else "stop"
                        )
                        yield self.stream_chunk(
                            usage={
                                "prompt_tokens": count,
                                "completion_tokens": 1,
                                "total_tokens": count + 1,
                                "prompt_tokens_details": {"cached_tokens": 7},
                            }
                        )
                        yield "data: [DONE]\n\n"

                    with patch.object(
                        self.chat, "_generate_chat_stream", upstream, create=True
                    ):
                        events = [
                            self.stream_event(frame)
                            async for frame in self.serving._generate_anthropic_stream(
                                None, chat, request, None
                            )
                        ]
                    self.assertEqual(
                        "".join(
                            event["delta"].get("text", "")
                            for event in events
                            if event["type"] == "content_block_delta"
                        ),
                        "" if truncated else "ANSWER",
                    )
                    self.assertEqual(
                        "".join(
                            event["delta"].get("thinking", "")
                            for event in events
                            if event["type"] == "content_block_delta"
                        ),
                        "private rationale" if visible else "",
                    )
                    self.assertNotIn("</think>", json.dumps(events))
                    self.assertNotIn("<think>", json.dumps(events))
                    self.assertNotIn("error", [event["type"] for event in events])
                    self.assertEqual(events[-2]["usage"], {"output_tokens": 1})
                    if truncated and not visible:
                        self.assertEqual(
                            [event["type"] for event in events],
                            ["message_start", "message_delta", "message_stop"],
                        )
                        usage = events[0]["message"]["usage"]
                        self.assertEqual(
                            usage["input_tokens"] + usage["cache_read_input_tokens"],
                            count,
                        )

    @bounded_cpu_test
    async def test_glm53_forced_json_tool_keeps_disabled_parser_path(self):
        self.use_glm53_template()
        self.chat_module.get_json_schema_constraint = sys.modules[
            "sglang.srt.function_call.utils"
        ].get_json_schema_constraint
        request = self.request(
            thinking={"type": "disabled"},
            tools=[
                {
                    "name": "lookup",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ],
            tool_choice={"type": "tool", "name": "lookup"},
        )
        chat = self.serving._convert_to_chat_completion_request(request)
        adapted, processed = self.serving._convert_to_internal_request(
            chat, None, request
        )
        self.assertEqual(processed._tool_call_constraint_kind, "json_schema")
        self.assertFalse(adapted.require_reasoning)
        self.assertFalse(processed.separate_reasoning)
        parsed = self.chat_module.Glm45Detector(force_reasoning=False).detect_and_parse(
            '[{"name":"lookup","parameters":{"city":"Hangzhou"}}]'
        )
        self.assertEqual(
            json.loads(parsed.normal_text)[0]["parameters"], {"city": "Hangzhou"}
        )
        self.assertEqual(parsed.reasoning_text, "")
        # Native tool markers already end reasoning even without </think>.
        native = self.chat_module.Glm45Detector(force_reasoning=True).detect_and_parse(
            "private<tool_call>lookup<arg_key>city</arg_key><arg_value>Hangzhou</arg_value></tool_call>"
        )
        self.assertEqual(native.reasoning_text, "private")
        self.assertTrue(native.normal_text.startswith("<tool_call>lookup"))

    @bounded_cpu_test
    async def test_hidden_reasoning_preserves_tool_blocks_and_arguments(self):
        for thinking in (
            {"type": "disabled"},
            {"type": "adaptive", "display": "omitted"},
        ):
            with self.subTest(thinking=thinking):
                request = self.request(thinking=thinking, stream=True)
                tool = {
                    "id": "call_lookup",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"city":"Hangzhou"}'},
                }
                response = self.chat_protocol.ChatCompletionResponse(
                    id="cpu",
                    created=0,
                    model="cpu-model",
                    usage={
                        "prompt_tokens": 37,
                        "completion_tokens": 12,
                        "total_tokens": 49,
                    },
                    choices=[
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "reasoning_content": "private",
                                "content": None,
                                "tool_calls": [tool],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                )
                converted = self.serving._convert_response(
                    response,
                    include_thinking=self.serving_module._include_thinking(request),
                )
                self.assertEqual(
                    [block.type for block in converted.content], ["tool_use"]
                )
                self.assertEqual(converted.content[0].input, {"city": "Hangzhou"})

                async def upstream(*args):
                    yield self.stream_chunk({"reasoning_content": "private"})
                    yield self.stream_chunk({"tool_calls": [{"index": 0, **tool}]})
                    yield self.stream_chunk({}, finish_reason="tool_calls")
                    yield "data: [DONE]\n\n"

                with patch.object(
                    self.chat, "_generate_chat_stream", upstream, create=True
                ):
                    events = [
                        self.stream_event(frame)
                        async for frame in self.serving._generate_anthropic_stream(
                            None, None, request, None
                        )
                    ]
                blocks = [
                    event["content_block"]
                    for event in events
                    if event["type"] == "content_block_start"
                ]
                self.assertEqual([block["type"] for block in blocks], ["tool_use"])
                arguments = "".join(
                    event["delta"].get("partial_json", "")
                    for event in events
                    if event["type"] == "content_block_delta"
                )
                self.assertEqual(json.loads(arguments), {"city": "Hangzhou"})
                self.assertNotIn("private", json.dumps(events))
                self.assertEqual(events[-2]["delta"]["stop_reason"], "tool_use")

    @bounded_cpu_test
    async def test_other_templates_keep_their_existing_reasoning_toggle(self):
        request = self.request(thinking={"type": "disabled"})
        chat = self.serving._convert_to_chat_completion_request(request)
        self.assertFalse(self.serving._uses_glm53_thinking_template())
        self.assertFalse(self.chat._get_reasoning_from_request(chat))

    @staticmethod
    def stream_chunk(delta=None, finish_reason=None, usage=None):
        return (
            "data: "
            + json.dumps(
                {
                    "id": "cpu-stream",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "cpu-model",
                    "choices": (
                        [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
                        if delta is not None
                        else []
                    ),
                    "usage": usage,
                }
            )
            + "\n\n"
        )

    @staticmethod
    def stream_event(frame):
        return json.loads(frame.split("data: ", 1)[1])

    def conversation(self):
        return self.request(
            system=[
                {"type": "text", "text": "first"},
                {
                    "type": "text",
                    "text": "second",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            thinking={"type": "enabled", "budget_tokens": 1024},
            output_config={"effort": "high"},
            tools=[
                {
                    "name": "read",
                    "input_schema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                }
            ],
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            messages=[
                {"role": "user", "content": "read file"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "inspect first"},
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "read",
                            "input": {"path": "a.txt"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "before"},
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [{"type": "text", "text": "contents"}],
                        },
                        {"type": "text", "text": "after"},
                    ],
                },
            ],
        )

    @bounded_cpu_test
    async def test_token_ids_and_counts_equal_chat_and_generation(self):
        for request in (self.request(), self.conversation()):
            with self.subTest(tools=bool(request.tools)):
                converted = self.serving._convert_to_chat_completion_request(request)
                expected = await self.worker.tokenize(
                    converted.model_dump(exclude_none=True)
                )
                self.assertTrue(expected["supported"], expected)
                response = await self.serving.handle_tokenize(
                    request.model_dump_json().encode(), None
                )
                self.assertEqual(json.loads(response.body), expected)
                count = await self.serving.handle_count_tokens(
                    self.protocol.AnthropicCountTokensRequest.model_validate(
                        request.model_dump()
                    ),
                    None,
                )
                self.assertEqual(
                    json.loads(count.body), {"input_tokens": len(expected["tokens"])}
                )
                processed = self.chat._process_messages(converted, True)
                key, prompt = self.chat._engine_prompt(processed, True)
                self.assertEqual(key, "text")
                self.assertEqual(expected["tokens"], self.tokenizer.encode(prompt))
                if request.tools:
                    for text in (
                        "TOOLS=",
                        "first",
                        "second",
                        "<think>inspect first",
                        "a.txt",
                        "contents",
                        "THINK=True",
                        "EFFORT=high",
                    ):
                        self.assertIn(text, prompt)
                    self.assertLess(prompt.index("before"), prompt.index("contents"))
                    self.assertLess(prompt.index("contents"), prompt.index("after"))
                    self.assertFalse(converted.parallel_tool_calls)

    @bounded_cpu_test
    async def test_metadata_reaches_both_internal_handlers(self):
        metadata = dict(
            bootstrap_host="prefill",
            bootstrap_port=7406,
            bootstrap_room=2**63 - 2,
            routed_dp_rank=0,
            disagg_prefill_dp_rank=1,
            cache_salt="tenant",
            rid="request-id",
            input_ids=[12, 34, 56],
        )
        raw = NS(headers={"x-data-parallel-rank": "2"})
        for stream in (False, True):
            for session in (
                {"session_id": "stable-session"},
                {
                    "session_params": {
                        "id": "s",
                        "rid": "previous",
                        "offset": 4,
                        "replace": True,
                        "drop_previous_output": False,
                    }
                },
            ):
                request = self.request(stream=stream, **metadata, **session)
                captured = []

                async def nonstream(adapted, processed, raw):
                    captured.append(adapted)
                    return self.chat_protocol.ChatCompletionResponse(
                        id="c",
                        created=0,
                        model="cpu-model",
                        choices=[],
                        usage={
                            "prompt_tokens": 3,
                            "completion_tokens": 0,
                            "total_tokens": 3,
                        },
                    )

                async def stream_gen(adapted, processed, anthropic, raw):
                    captured.append(adapted)
                    yield "data: done\n\n"

                with (
                    patch.object(
                        self.chat,
                        "_handle_non_streaming_request",
                        nonstream,
                        create=True,
                    ),
                    patch.object(
                        self.serving, "_generate_anthropic_stream", stream_gen
                    ),
                ):
                    response = await self.serving.handle_messages(request, raw)
                    self.assertEqual(
                        response.status_code, 200, getattr(response, "body", "")
                    )
                    if stream:
                        _ = [chunk async for chunk in response.body_iterator]
                self.assertEqual(len(captured), 1)
                for key, value in {**metadata, **session}.items():
                    self.assertEqual(
                        getattr(captured[0], key),
                        2 if key == "routed_dp_rank" else value,
                        key,
                    )

    @bounded_cpu_test
    async def test_metadata_types_unknown_beta_fields_and_sessions(self):
        request = self.request(
            bootstrap_host=["p0", "p1"],
            bootstrap_port=[7406, None],
            bootstrap_room=[0, 2**63 - 1],
            cache_salt=["a", "b"],
            context_management={"edits": []},
            future_beta=True,
            custom_logit_processor="must not forward",
            priority=99,
        )
        converted = self.serving._convert_to_chat_completion_request(request)
        for field in (
            "bootstrap_host",
            "bootstrap_port",
            "bootstrap_room",
            "cache_salt",
        ):
            self.assertEqual(getattr(converted, field), getattr(request, field))
        self.assertIsNone(converted.custom_logit_processor)
        self.assertIsNone(converted.priority)
        for bad in ([-1], [True], ["1"], [2**32]):
            with self.assertRaises(ValueError):
                self.request(input_ids=bad)
        with self.assertRaisesRegex(ValueError, "cannot both"):
            self.request(session_id="s", session_params={"id": "s"})

    @bounded_cpu_test
    async def test_pretokenized_ids_survive_text_only(self):
        request = self.conversation().model_copy(update={"input_ids": [11, 22, 33]})
        response = await self.serving.handle_tokenize(request, None)
        body = json.loads(response.body)
        self.assertTrue(body["supported"], body)
        self.assertEqual(body["tokens"], [11, 22, 33])
        self.assertEqual(body["tokenization_path"], "request_input_ids")

    @bounded_cpu_test
    async def test_images_never_reuse_prompt_ids_or_claim_exact_cpu_counts(self):
        self.chat.template_manager.jinja_template_content_format = "openai"
        for source in (
            {"type": "base64", "media_type": "image/png", "data": "eA=="},
            {"type": "url", "url": "https://example.invalid/image.png"},
        ):
            for tool_result in (False, True):
                block = {"type": "image", "source": source}
                if tool_result:
                    block = {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [block],
                    }
                request = self.request(
                    input_ids=[999], messages=[{"role": "user", "content": [block]}]
                )
                converted = self.serving._convert_to_chat_completion_request(request)
                self.assertIsNone(converted.input_ids)
                adapted, _ = self.serving._convert_to_internal_request(converted, None)
                self.assertTrue(adapted.image_data)
                self.assertNotIn("input_ids", vars(adapted))
                response = await self.serving.handle_tokenize(request, None)
                self.assertEqual(
                    json.loads(response.body),
                    {"supported": False, "reason": "unsupported_request"},
                )
                count = await self.serving.handle_count_tokens(
                    self.protocol.AnthropicCountTokensRequest.model_validate(
                        request.model_dump()
                    ),
                    None,
                )
                self.assertEqual(count.status_code, 400)
                self.assertEqual(json.loads(count.body)["type"], "error")

    @bounded_cpu_test
    async def test_bad_images_and_validation_errors_are_anthropic_errors(self):
        for payload in (
            b"{",
            b"{}",
            self.request()
            .model_dump_json()
            .replace('"max_tokens":32', '"max_tokens":0')
            .encode(),
            json.dumps(
                {
                    "model": "cpu-model",
                    "max_tokens": 32,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "source": {"type": "base64"}}
                            ],
                        }
                    ],
                }
            ).encode(),
        ):
            response = await self.serving.handle_tokenize(payload, None)
            self.assertEqual(response.status_code, 400)
            self.assertEqual(
                json.loads(response.body)["error"]["type"], "invalid_request_error"
            )

    @bounded_cpu_test
    async def test_shared_worker_bound_and_conversion_off_event_loop(self):
        event_loop_thread = threading.get_ident()
        seen = []
        original = self.serving._convert_to_chat_completion_request

        def convert(request):
            seen.append(threading.get_ident())
            return original(request)

        with patch.object(self.serving, "_convert_to_chat_completion_request", convert):
            response = await self.serving.handle_tokenize(self.request(), None)
        self.assertTrue(json.loads(response.body)["supported"])
        self.assertNotEqual(seen, [event_loop_thread])
        await self.worker._slot.acquire()
        try:
            response = await self.serving.handle_tokenize(self.request(), None)
            self.assertEqual(
                json.loads(response.body), {"supported": False, "reason": "busy"}
            )
            count = await self.serving.handle_count_tokens(
                self.protocol.AnthropicCountTokensRequest.model_validate(
                    self.request().model_dump()
                ),
                None,
            )
            self.assertEqual(count.status_code, 503)
        finally:
            self.worker._slot.release()

    @bounded_cpu_test
    async def test_http_routes_match_chat_and_use_anthropic_error_envelopes(self):
        from http import HTTPStatus

        import httpx
        from fastapi import Depends, FastAPI, HTTPException, Request
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import ORJSONResponse

        # Execute the actual route and exception-handler declarations. Lifespan
        # startup is deliberately omitted: ASGITransport performs in-memory IO.
        app = FastAPI()
        app.state.chat_affinity_tokenizer = self.worker
        app.state.anthropic_serving = self.serving
        ns = dict(
            app=app,
            HTTPStatus=HTTPStatus,
            Depends=Depends,
            HTTPException=HTTPException,
            Request=Request,
            RequestValidationError=RequestValidationError,
            ORJSONResponse=ORJSONResponse,
            ErrorResponse=self.chat_protocol.ErrorResponse,
            AnthropicCountTokensRequest=self.protocol.AnthropicCountTokensRequest,
            logger=logging.getLogger(__name__),
        )
        names = {
            "_anthropic_validation_message",
            "_anthropic_error_response",
            "validation_exception_handler",
            "validate_json_request",
            "chat_affinity_tokenize",
            "anthropic_v1_tokenize",
            "anthropic_v1_count_tokens",
        }
        path = ENTRY / "http_server.py"
        tree = ast.parse(path.read_text())
        tree.body = [n for n in tree.body if getattr(n, "name", None) in names]
        self.assertEqual(len(tree.body), len(names) + 1)  # two error handlers
        exec(compile(tree, str(path), "exec"), ns)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://cpu.test"
        ) as client:
            request = self.conversation()
            messages = await client.post(
                "/v1/messages/tokenize", json=request.model_dump(exclude_none=True)
            )
            chat = await client.post(
                "/v1/chat/tokenize",
                json=self.serving._convert_to_chat_completion_request(
                    request
                ).model_dump(exclude_none=True),
            )
            self.assertEqual(messages.status_code, 200)
            self.assertEqual(messages.json(), chat.json())
            count = await client.post(
                "/v1/messages/count_tokens", json=request.model_dump(exclude_none=True)
            )
            self.assertEqual(count.json(), {"input_tokens": messages.json()["count"]})
            for content, headers in (
                (b"{}", {}),
                (b"{", {"content-type": "application/json"}),
                (b"{}", {"content-type": "application/json"}),
            ):
                error = await client.post(
                    "/v1/messages/tokenize", content=content, headers=headers
                )
                self.assertEqual(error.status_code, 400)
                self.assertEqual(error.json()["type"], "error")
                self.assertEqual(error.json()["error"]["type"], "invalid_request_error")
            with patch.object(self.affinity, "MAX_REQUEST_BYTES", 8):
                too_large = await client.post(
                    "/v1/messages/tokenize", json={"long": "body"}
                )
                self.assertEqual(
                    too_large.json(), {"supported": False, "reason": "too_large"}
                )

    @bounded_cpu_test
    async def test_stream_empty_finish_retains_latest_full_usage(self):
        usage = {
            "prompt_tokens": 37,
            "completion_tokens": 1,
            "total_tokens": 38,
            "prompt_tokens_details": {"cached_tokens": 29},
        }
        final = self.stream_chunk(usage=usage)
        finish = self.stream_chunk({}, finish_reason="length")
        earlier = self.stream_chunk(
            usage={
                "prompt_tokens": 37,
                "completion_tokens": 0,
                "total_tokens": 37,
                "prompt_tokens_details": {"cached_tokens": 10},
            }
        )
        for chunks in ([finish, final], [earlier, finish, final], [final, finish]):
            with self.subTest(chunks=chunks):
                consumed = []
                stream = self.stream([*chunks, "data: [DONE]\n\n"], consumed)
                first = self.stream_event(await anext(stream))
                self.assertIn(final, consumed)
                self.assertEqual(first["type"], "message_start")
                self.assertEqual(
                    first["message"]["usage"],
                    {
                        "input_tokens": 8,
                        "cache_read_input_tokens": 29,
                        "output_tokens": 0,
                    },
                )
                rest = [self.stream_event(frame) async for frame in stream]
                self.assertEqual(
                    [event["type"] for event in rest], ["message_delta", "message_stop"]
                )
                self.assertEqual(rest[0]["delta"]["stop_reason"], "max_tokens")
                self.assertEqual(rest[0]["usage"], {"output_tokens": 1})

    @bounded_cpu_test
    async def test_stream_zero_argument_tool_first_delta_is_immediate(self):
        for arguments in (None, ""):
            with self.subTest(arguments=arguments):
                function = {"name": "get_status"}
                if arguments is not None:
                    function["arguments"] = arguments
                first_chunk = self.stream_chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_status",
                                "type": "function",
                                "function": function,
                            }
                        ]
                    }
                )
                consumed = []
                stream = self.stream(
                    [
                        first_chunk,
                        self.stream_chunk({}, finish_reason="tool_calls"),
                        "data: [DONE]\n\n",
                    ],
                    consumed,
                )
                start = self.stream_event(await anext(stream))
                self.assertEqual(start["type"], "message_start")
                block = self.stream_event(await anext(stream))
                self.assertEqual(consumed, [first_chunk])
                self.assertEqual(block["type"], "content_block_start")
                self.assertEqual(
                    block["content_block"],
                    {
                        "type": "tool_use",
                        "id": "call_status",
                        "name": "get_status",
                        "input": {},
                    },
                )
                rest = [self.stream_event(frame) async for frame in stream]
                self.assertEqual(
                    [event["type"] for event in rest],
                    [
                        "content_block_stop",
                        "message_delta",
                        "message_stop",
                    ],
                )
                self.assertEqual(rest[1]["delta"]["stop_reason"], "tool_use")

    @bounded_cpu_test
    async def test_stream_first_text_or_thinking_does_not_wait_for_usage(self):
        for delta, kind in (
            ({"content": "first"}, "text"),
            ({"reasoning_content": "first"}, "thinking"),
        ):
            with self.subTest(kind=kind):
                first_chunk = self.stream_chunk(delta)
                consumed = []
                stream = self.stream(
                    [
                        first_chunk,
                        self.stream_chunk({}, finish_reason="stop"),
                        self.stream_chunk(
                            usage={
                                "prompt_tokens": 37,
                                "completion_tokens": 1,
                                "total_tokens": 38,
                            }
                        ),
                        "data: [DONE]\n\n",
                    ],
                    consumed,
                )
                events = [self.stream_event(await anext(stream)) for _ in range(3)]
                self.assertEqual(consumed, [first_chunk])
                self.assertEqual(
                    [event["type"] for event in events],
                    [
                        "message_start",
                        "content_block_start",
                        "content_block_delta",
                    ],
                )
                self.assertEqual(events[1]["content_block"]["type"], kind)
                self.assertEqual(events[2]["delta"][kind], "first")
                rest = [self.stream_event(frame) async for frame in stream]
                self.assertEqual(
                    [event["type"] for event in rest],
                    [
                        "content_block_stop",
                        "message_delta",
                        "message_stop",
                    ],
                )
                self.assertEqual(rest[1]["usage"], {"output_tokens": 1})

    @bounded_cpu_test
    async def test_worker_completion_when_cross_thread_wakeup_is_denied(self):
        loop = asyncio.get_running_loop()
        original = self.serving._convert_to_chat_completion_request

        def convert(request):
            # Ensure the awaiter is installed before the worker completes.
            time.sleep(0.02)
            return original(request)

        # asyncio suppresses the EPERM from its socket send, leaving the ready
        # callback queued. Reproduce that dropped wake-up even outside a sandbox.
        with (
            patch.object(loop, "_write_to_self") as wakeup,
            patch.object(self.serving, "_convert_to_chat_completion_request", convert),
        ):
            response = await self.serving.handle_tokenize(self.request(), None)
            self.assertTrue(json.loads(response.body)["supported"])
            self.assertTrue(wakeup.called)
        self.assertFalse(self.worker._slot.locked())

    @bounded_cpu_test
    async def test_deadline_cancels_stalled_await_and_dumps_threads(self):
        cancelled = asyncio.Event()

        async def stalled():
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        with patch.object(faulthandler, "dump_traceback") as dump:
            with self.assertRaises(TimeoutError):
                await run_bounded_cpu_test(stalled(), timeout=0.03)
            dump.assert_called_once_with()
        self.assertTrue(cancelled.is_set())

    @bounded_cpu_test
    async def test_cancellation_keeps_shared_slot_until_worker_finishes(self):
        started = threading.Event()
        finish = threading.Event()
        original = self.serving._convert_to_chat_completion_request

        def convert(request):
            started.set()
            if not finish.wait(timeout=5):
                raise AssertionError("test did not release the CPU worker")
            return original(request)

        with patch.object(self.serving, "_convert_to_chat_completion_request", convert):
            task = asyncio.create_task(
                self.serving.handle_tokenize(self.request(), None)
            )
            try:
                for _ in range(100):
                    if started.is_set():
                        break
                    await asyncio.sleep(0.01)
                self.assertTrue(started.is_set())
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertTrue(self.worker._slot.locked())
                busy = await self.worker.tokenize({})
                self.assertEqual(busy, {"supported": False, "reason": "busy"})
            finally:
                finish.set()
                for _ in range(100):
                    if not self.worker._slot.locked():
                        break
                    await asyncio.sleep(0.01)
        self.assertFalse(self.worker._slot.locked())


if __name__ == "__main__":
    unittest.main()
