"""Opt-in, per-render proof for eliminating GLM5Next's final text encode.

This does not assume that encode(decode(ids)) is an identity. The unchanged
renderer must supply a witness that its first input text equals the final input
text, with the same deterministic encoder/options. Unknown cases return None.

In ChatAffinityTokenizer, construct Glm53ChatTokenizationGuard immediately before
_process_messages, after installing the empty request-local template cache. Call
try_reuse immediately afterwards. Keep the complete existing fallback and the
existing final model/tokenizer identity checks. Never share a guard across calls.
No SGLang, Transformers, Torch, or GPU modules are imported by this module.
"""

import json
import os
import sys
from collections import OrderedDict

OPT_IN_ENV = "SGLANG_GLM53_CHAT_REUSE_PROMPT_IDS"

# Scalar encode and batch __call__ both enter _encode_plus -> encode_batch in
# these inspected implementations. 5.12.1's BPE cleanup change is unreachable
# with the cleanup-disabled profile below. Unknown versions fail closed.
_AUDITED_VERSIONS = frozenset({("5.6.0", "0.22.2"), ("5.12.1", "0.22.2")})
_SERVING_MODULE = "sglang.srt.entrypoints.openai.serving_chat"
_TOKENIZER_MODULE = "transformers.tokenization_utils_tokenizers"
_GLM53_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
_EXPECTED_PRE = {
    "type": "Sequence",
    "pretokenizers": [
        {
            "type": "Split",
            "pattern": {"Regex": _GLM53_SPLIT},
            "behavior": "Isolated",
            "invert": False,
        },
        {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": False,
        },
    ],
}
_EXPECTED_POST = {
    "type": "ByteLevel",
    "add_prefix_space": True,
    "trim_offsets": False,
    "use_regex": True,
}
_EXPECTED_DECODER = {
    "type": "ByteLevel",
    "add_prefix_space": True,
    "trim_offsets": True,
    "use_regex": True,
}
_RENDER_METHODS = (
    "_process_messages",
    "_apply_jinja_template",
    "_render_and_encode_chat_template",
    "_encode_messages",
    "_engine_prompt",
    "_handle_last_assistant_message",
    "_append_assistant_prefix_to_prompt_ids",
)


def _state(component):
    return None if component is None else json.loads(component.__getstate__())


def _tokenizer_profile(tokenizer):
    """Return a cheap live profile, never serialize the full vocabulary/merges."""
    transformers = sys.modules.get("transformers")
    tokenizers = sys.modules.get("tokenizers")
    module = sys.modules.get(_TOKENIZER_MODULE)
    if (
        getattr(transformers, "__version__", None),
        getattr(tokenizers, "__version__", None),
    ) not in _AUDITED_VERSIONS or type(tokenizer) is not getattr(
        module, "TokenizersBackend", None
    ):
        return None
    if (
        tokenizer.is_fast is not True
        or tokenizer.clean_up_tokenization_spaces is not False
        or tokenizer.split_special_tokens is not False
        or tokenizer.bos_token_id is not None
        or getattr(tokenizer, "add_bos_token", False)
        or getattr(tokenizer, "add_eos_token", False)
        or getattr(tokenizer, "_in_target_context_manager", False)
        # Reject instance-level encoder/decode/hooks as well as subclasses.
        or any(callable(value) for value in vars(tokenizer).values())
    ):
        return None
    backend = tokenizer.backend_tokenizer
    if (
        type(backend) is not tokenizers.Tokenizer
        or type(backend.model) is not tokenizers.models.BPE
        or backend.model.dropout is not None
        or backend.normalizer is not None
        or backend.padding is not None
        or backend.truncation is not None
        or backend.encode_special_tokens
        or backend.model.unk_token is not None
        or backend.model.continuing_subword_prefix is not None
        or backend.model.end_of_word_suffix is not None
        or backend.model.fuse_unk
        or backend.model.byte_fallback
        or backend.model.ignore_merges is not True
        or backend.num_special_tokens_to_add(False) != 0
    ):
        return None
    pre = _state(backend.pre_tokenizer)
    post = _state(backend.post_processor)
    decoder = _state(backend.decoder)
    if (
        pre != _EXPECTED_PRE
        # This exact ByteLevel postprocessor only changes offsets, never IDs.
        or post != _EXPECTED_POST
        or decoder != _EXPECTED_DECODER
    ):
        return None
    added = backend.get_added_tokens_decoder()
    if len(added) != 36 or any(t.lstrip or t.rstrip for t in added.values()):
        return None
    added_profile = tuple(
        (i, t.content, t.single_word, t.lstrip, t.rstrip, t.normalized, t.special)
        for i, t in sorted(added.items())
    )
    return backend, pre, post, decoder, added_profile


def _text_only(request):
    if (
        request.input_ids is not None
        or request.continue_final_message
        or request.n not in (None, 1)
        or any(
            getattr(request, name, None) is not None
            for name in (
                "lora_path",
                "session_id",
                "session_params",
                "input_embeds",
                "image_data",
                "video_data",
                "audio_data",
                "mm_hashes",
            )
        )
        or not request.messages
    ):
        return False
    for message in request.messages:
        content = message.content
        if content is None or isinstance(content, str):
            continue
        if not isinstance(content, list):
            return False
        for part in content:
            kind = (
                part.get("type")
                if isinstance(part, dict)
                else getattr(part, "type", None)
            )
            text = (
                part.get("text")
                if isinstance(part, dict)
                else getattr(part, "text", None)
            )
            if kind != "text" or not isinstance(text, str):
                return False
    return True


def _ordinary_glm53_renderer(serving, request, is_multimodal):
    module = sys.modules.get(_SERVING_MODULE)
    cls = getattr(module, "OpenAIServingChat", None)
    if cls is None or type(serving) is not cls:
        return False
    if any(
        getattr(getattr(serving, name, None), "__func__", None)
        is not getattr(cls, name, None)
        for name in _RENDER_METHODS
    ):
        return False
    manager = serving.tokenizer_manager
    config = manager.model_config.hf_config
    return (
        is_multimodal is True
        and manager.model_config.is_multimodal is True
        and config.model_type == "glm5_next"
        and config.architectures == ["Glm5NextForConditionalGeneration"]
        and request.model in (manager.served_model_name, "default")
        and serving.chat_encoding_spec is None
        and not serving.is_gpt_oss
        and not serving.is_gemma4
        and serving.template_manager.chat_template_name is None
        and serving._tokenizer_auto_adds_specials is False
        and isinstance(manager.tokenizer.chat_template, str)
        and _text_only(request)
    )


class Glm53ChatTokenizationGuard:
    """Single-use proof using only the caller's fresh request-local render cache.

    Tokenizer/model configuration must be immutable during a render, as required
    by the reference path too. Existing outer identity-after checks are required.
    The guard observes configuration before and after; it does not lock or mutate
    the shared tokenizer. It never retains token results across requests.
    """

    def __init__(self, serving, request, is_multimodal):
        self._ready = False
        if os.environ.get(OPT_IN_ENV) != "1":
            return
        try:
            cache = serving._chat_template_cache
            if (
                type(cache) is not OrderedDict
                or cache
                or not _ordinary_glm53_renderer(serving, request, is_multimodal)
            ):
                return
            tokenizer = serving.tokenizer_manager.tokenizer
            profile = _tokenizer_profile(tokenizer)
            if profile is None:
                return
            self._serving = serving
            self._request = request
            self._cache = cache
            self._tokenizer = tokenizer
            self._profile = profile
            self._ready = True
        except (AttributeError, TypeError, ValueError, RuntimeError):
            # Optional optimization: unsupported introspection must fall back.
            return

    def try_reuse(self, processed):
        """Return the first IDs only with a complete witness, otherwise None."""
        if not self._ready:
            return None
        self._ready = False
        try:
            serving = self._serving
            if (
                os.environ.get(OPT_IN_ENV) != "1"
                or serving.tokenizer_manager.tokenizer is not self._tokenizer
                or serving._chat_template_cache is not self._cache
                or len(self._cache) != 1
                or not _ordinary_glm53_renderer(serving, self._request, True)
                or _tokenizer_profile(self._tokenizer) != self._profile
                or processed.image_data
                or processed.video_data
                or processed.audio_data
                or processed.modalities
            ):
                return None
            rendered, ids, decoded = next(iter(self._cache.values()))
            if (
                type(rendered) is not str
                or not rendered
                or type(decoded) is not str
                or type(processed.prompt) is not str
                or rendered != decoded
                or rendered != processed.prompt
                or type(ids) is not tuple
                or type(processed.prompt_ids) is not list
                or not ids
                or ids != tuple(processed.prompt_ids)
            ):
                return None
            # Same exact text + same deterministic encoder + same default options.
            # Caller retains MAX_TOKENS/range validation and model identity checks.
            return processed.prompt_ids
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return None
