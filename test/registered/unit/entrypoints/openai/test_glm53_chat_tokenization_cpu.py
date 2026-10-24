"""Dependency-free, small CPU tests for the opt-in per-render proof."""

import importlib.util
import os
import sys
import types
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[5]
PATH = ROOT / "python/sglang/srt/entrypoints/openai/glm53_chat_tokenization.py"
SPEC = importlib.util.spec_from_file_location("glm53_chat_tokenization_cpu", PATH)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)
NS = types.SimpleNamespace


class OpenAIServingChat:
    pass


def _unused(self, *args, **kwargs):
    raise AssertionError("the proof helper must never call the renderer/encoder")


for _name in helper._RENDER_METHODS:
    setattr(OpenAIServingChat, _name, _unused)


class TestGlm53ChatTokenizationGuard(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {helper.OPT_IN_ENV: "1"}))
        self.enterContext(
            patch.dict(
                sys.modules,
                {helper._SERVING_MODULE: NS(OpenAIServingChat=OpenAIServingChat)},
            )
        )
        self.profile = self.enterContext(
            patch.object(helper, "_tokenizer_profile", return_value=("audited",))
        )
        self.serving = OpenAIServingChat()
        self.serving.tokenizer_manager = NS(
            tokenizer=NS(chat_template="actual template"),
            served_model_name="glm",
            model_config=NS(
                is_multimodal=True,
                hf_config=NS(
                    model_type="glm5_next",
                    architectures=["Glm5NextForConditionalGeneration"],
                ),
            ),
        )
        self.serving.template_manager = NS(chat_template_name=None)
        self.serving.chat_encoding_spec = None
        self.serving.is_gpt_oss = self.serving.is_gemma4 = False
        self.serving._tokenizer_auto_adds_specials = False
        self.serving._chat_template_cache = OrderedDict()
        self.request = NS(
            model="glm",
            input_ids=None,
            n=1,
            continue_final_message=False,
            messages=[NS(content="hello")],
        )
        self.processed = NS(
            prompt="rendered",
            prompt_ids=[1, 2, 3],
            image_data=None,
            audio_data=None,
            video_data=None,
            modalities=[],
        )

    def begin(self, multimodal=True):
        return helper.Glm53ChatTokenizationGuard(self.serving, self.request, multimodal)

    def witness(self):
        self.serving._chat_template_cache[b"current render"] = (
            "rendered",
            (1, 2, 3),
            "rendered",
        )

    def test_default_off_and_only_explicit_one_enables(self):
        for value in (None, "", "0", "true", "yes", "2"):
            with self.subTest(value=value), patch.dict(os.environ):
                if value is None:
                    os.environ.pop(helper.OPT_IN_ENV, None)
                else:
                    os.environ[helper.OPT_IN_ENV] = value
                guard = self.begin()
                self.assertIsNone(guard.try_reuse(self.processed))
        self.profile.assert_not_called()

    def test_exact_witness_returns_first_ids_and_is_single_use(self):
        guard = self.begin()
        self.witness()
        self.assertIs(guard.try_reuse(self.processed), self.processed.prompt_ids)
        self.assertIsNone(guard.try_reuse(self.processed))

    def test_mismatched_text_or_ids_always_falls_back(self):
        for field, value in (
            ("prompt", "rendered "),
            ("prompt", "rendered\ufffd"),
            ("prompt", ""),
            ("prompt_ids", [1, 2, 4]),
            ("prompt_ids", [1, 2, 3, 4]),
            ("prompt_ids", (1, 2, 3)),
        ):
            with self.subTest(field=field, value=value):
                self.serving._chat_template_cache.clear()
                guard = self.begin()
                self.witness()
                with patch.object(self.processed, field, value):
                    self.assertIsNone(guard.try_reuse(self.processed))

    def test_decode_cleanup_cannot_be_inferred_from_first_ids(self):
        guard = self.begin()
        self.serving._chat_template_cache[b"current render"] = (
            "hello .",
            (1, 2, 3),
            "hello.",
        )
        self.processed.prompt = "hello."
        self.assertIsNone(guard.try_reuse(self.processed))

    def test_prepopulated_replaced_missing_or_multiple_cache_falls_back(self):
        self.witness()
        self.assertIsNone(self.begin().try_reuse(self.processed))
        self.serving._chat_template_cache.clear()
        guard = self.begin()
        self.witness()
        self.serving._chat_template_cache = self.serving._chat_template_cache.copy()
        self.assertIsNone(guard.try_reuse(self.processed))
        self.serving._chat_template_cache.clear()
        self.assertIsNone(self.begin().try_reuse(self.processed))
        guard = self.begin()
        self.witness()
        self.serving._chat_template_cache[b"other"] = (
            "rendered",
            (1, 2, 3),
            "rendered",
        )
        self.assertIsNone(guard.try_reuse(self.processed))

    def test_unsupported_renderers_and_models(self):
        for obj, name, value in (
            (self.serving, "chat_encoding_spec", "custom"),
            (self.serving, "is_gpt_oss", True),
            (self.serving, "is_gemma4", True),
            (self.serving, "_tokenizer_auto_adds_specials", True),
            (self.serving.template_manager, "chat_template_name", "named"),
            (self.serving.tokenizer_manager.tokenizer, "chat_template", {}),
            (
                self.serving.tokenizer_manager.model_config.hf_config,
                "model_type",
                "other",
            ),
            (
                self.serving.tokenizer_manager.model_config.hf_config,
                "architectures",
                [],
            ),
            (self.serving.tokenizer_manager.model_config, "is_multimodal", False),
            (self.request, "model", "other"),
            (self.request, "input_ids", [1]),
            (self.request, "continue_final_message", True),
            (self.request, "n", 2),
        ):
            with self.subTest(name=name), patch.object(obj, name, value):
                self.assertFalse(self.begin()._ready)
        self.assertFalse(self.begin(False)._ready)
        self.serving._process_messages = _unused
        self.assertFalse(self.begin()._ready)
        del self.serving._process_messages
        self.serving.__class__ = type("Subclass", (OpenAIServingChat,), {})
        self.assertFalse(self.begin()._ready)

    def test_input_media_and_session_extensions_reject(self):
        for content in (
            [{"type": "image_url", "image_url": {"url": "x"}}],
            [NS(type="video_url")],
            [{"type": "audio_url"}],
            [{"type": "text", "text": 4}],
            {"text": "x"},
        ):
            with self.subTest(content=content):
                self.request.messages = [NS(content=content)]
                self.assertFalse(self.begin()._ready)
        self.request.messages = [NS(content="x")]
        for name in (
            "lora_path",
            "session_id",
            "input_embeds",
            "image_data",
            "mm_hashes",
        ):
            with self.subTest(name=name), patch.object(
                self.request, name, [], create=True
            ):
                self.assertFalse(self.begin()._ready)

    def test_text_parts_and_tool_argument_urls_are_eligible(self):
        self.request.messages = [
            NS(
                content=None,
                tool_calls=[{"function": {"arguments": '{"image_url":"x"}'}}],
            ),
            NS(content=[{"type": "text", "text": "a"}, NS(type="text", text="b")]),
        ]
        self.assertTrue(self.begin()._ready)

    def test_processed_media_or_modalities_reject(self):
        for name in ("image_data", "video_data", "audio_data", "modalities"):
            with self.subTest(name=name):
                self.serving._chat_template_cache.clear()
                guard = self.begin()
                self.witness()
                with patch.object(self.processed, name, ["media"]):
                    self.assertIsNone(guard.try_reuse(self.processed))

    def test_live_tokenizer_profile_changes_and_opt_out_reject(self):
        guard = self.begin()
        self.witness()
        self.profile.return_value = ("changed",)
        self.assertIsNone(guard.try_reuse(self.processed))
        self.serving._chat_template_cache.clear()
        guard = self.begin()
        self.witness()
        self.serving.tokenizer_manager.tokenizer = NS(chat_template="actual template")
        self.assertIsNone(guard.try_reuse(self.processed))
        self.serving._chat_template_cache.clear()
        guard = self.begin()
        self.witness()
        os.environ[helper.OPT_IN_ENV] = "0"
        self.assertIsNone(guard.try_reuse(self.processed))

    def test_unknown_profile_and_introspection_failures_reject(self):
        self.profile.return_value = None
        self.assertFalse(self.begin()._ready)
        self.profile.side_effect = ValueError("unknown profile")
        self.assertFalse(self.begin()._ready)
        self.assertFalse(helper.Glm53ChatTokenizationGuard(None, None, True)._ready)


class TestGlm53VersionGate(unittest.TestCase):
    def test_unreviewed_library_pairs_fail_before_tokenizer_access(self):
        class TokenizersBackend:
            def __getattribute__(self, name):
                raise AssertionError("unreviewed runtime must not enter profile checks")

        for transformers_version, tokenizers_version in (
            ("5.12.0", "0.22.2"),
            ("5.12.2", "0.22.2"),
            ("5.12.1", "0.22.3"),
            ("5.6.0", "0.22.1"),
        ):
            with self.subTest(pair=(transformers_version, tokenizers_version)):
                with patch.dict(
                    sys.modules,
                    {
                        "transformers": NS(__version__=transformers_version),
                        "tokenizers": NS(__version__=tokenizers_version),
                        helper._TOKENIZER_MODULE: NS(
                            TokenizersBackend=TokenizersBackend
                        ),
                    },
                ):
                    self.assertIsNone(helper._tokenizer_profile(TokenizersBackend()))


if __name__ == "__main__":
    unittest.main()
