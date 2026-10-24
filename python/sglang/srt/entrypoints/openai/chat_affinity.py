"""CPU-only advisory tokenization for HTTP PD routing.

No scheduler requests, media processors, token cache, or incremental template
rendering. Re-render the complete history: a new user turn can change old
reasoning via clear_thinking. The serving instance (including its tokenizer and
model version) is consulted on every call.
"""

import asyncio
import copy
import json
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_TOKENS = 1024 * 1024


def supports_text_request(data):
    if not isinstance(data, dict) or data.get("n", 1) not in (None, 1):
        return False
    if any(
        data.get(k) is not None
        for k in (
            "lora_path",
            "session_id",
            "session_params",
            "input_embeds",
            "image_data",
            "video_data",
            "audio_data",
            "mm_hashes",
        )
    ):
        return False
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    for message in messages:
        if not isinstance(message, dict):
            return False
        content = message.get("content")
        if content is None or isinstance(content, str):
            continue
        if not isinstance(content, list) or any(
            not isinstance(part, dict)
            or part.get("type") != "text"
            or not isinstance(part.get("text"), str)
            for part in content
        ):
            return False
    return True


class ChatAffinityTokenizer:
    def __init__(self, chat_serving):
        self.chat_serving = chat_serving
        # Each render owns its cache and request copy. Keep CPU work bounded,
        # but allow deployments with long simultaneous histories to avoid
        # falling back simply because another advisory render is in flight.
        try:
            workers = int(os.environ.get("SGLANG_CHAT_AFFINITY_CPU_WORKERS", "1"))
        except ValueError:
            workers = 1
        workers = max(1, min(workers, 8))
        self._executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="chat-affinity"
        )
        self._slot = asyncio.Semaphore(workers)

    async def tokenize(self, data):
        # Never queue an unbounded amount of CPU work behind inference's HTTP
        # loop. A cancelled HTTP request must not free the slot before its CPU
        # job finishes (run_in_executor cannot interrupt Jinja/tokenizers).
        if self._slot.locked():
            return {"supported": False, "reason": "busy"}
        await self._slot.acquire()
        loop = asyncio.get_running_loop()
        try:
            future = loop.run_in_executor(self._executor, self._tokenize, data)
        except BaseException:
            self._slot.release()
            raise

        def finished(done):
            self._slot.release()
            if not done.cancelled():
                done.exception()  # Also retrieve failures after HTTP cancellation.

        future.add_done_callback(finished)
        return await asyncio.shield(future)

    def close(self):
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _tokenize(self, data):
        if isinstance(data, bytes):
            data = json.loads(data)
        if not supports_text_request(data):
            return {"supported": False, "reason": "unsupported_request"}
        # Isolate per-render caches from the event-loop serving object. Use the
        # very same implementation and loaded tokenizer, never a second model.
        serving = copy.copy(self.chat_serving)
        serving._chat_template_cache = OrderedDict()
        manager = serving.tokenizer_manager
        from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

        if (
            type(serving) is not OpenAIServingChat
            or serving.chat_encoding_spec is not None
            or serving.is_gpt_oss
            or serving.template_manager.chat_template_name is not None
            or manager.tokenizer is None
            or "MossVLForConditionalGeneration"
            in (getattr(manager.model_config.hf_config, "architectures", None) or [])
        ):
            return {"supported": False, "reason": "unsupported_renderer"}
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        tokenizer_before = manager.tokenizer
        identity_before = (
            manager.served_model_name,
            manager.model_path,
            manager.config_value("weight_version"),
        )
        request = ChatCompletionRequest.model_validate(copy.deepcopy(data))
        # Adapter model names can select a different tokenizer/cache namespace.
        if request.model not in (manager.served_model_name, "default"):
            return {"supported": False, "reason": "model_mismatch"}
        error = serving._validate_request(request)
        if error:
            return {"supported": False, "reason": "invalid_request"}
        serving._normalize_reasoning_effort(request)
        is_multimodal = manager.model_config.is_multimodal
        processed = serving._process_messages(request, is_multimodal)
        if processed.image_data or processed.video_data or processed.audio_data:
            return {"supported": False, "reason": "media"}
        if request.input_ids is not None:
            tokens = processed.prompt_ids
        else:
            kind, prompt = serving._engine_prompt(processed, is_multimodal)
            if kind == "input_ids":
                tokens = prompt
            elif isinstance(prompt, str) and prompt:
                # Match TokenizerManager._tokenize_texts' ordinary text path,
                # including the VLM decode/re-encode and default special tokens.
                tokenizer = manager.tokenizer
                tokens = (
                    tokenizer([prompt])["input_ids"][0]
                    if getattr(tokenizer, "is_fast", False)
                    else tokenizer.encode(prompt)
                )
            else:
                return {"supported": False, "reason": "unsupported_prompt"}
        if (
            not isinstance(tokens, list)
            or not 0 < len(tokens) <= MAX_TOKENS
            or any(type(t) is not int or not 0 <= t < 2**32 for t in tokens)
        ):
            return {"supported": False, "reason": "invalid_tokens"}
        identity_after = (
            manager.served_model_name,
            manager.model_path,
            manager.config_value("weight_version"),
        )
        if (
            manager.tokenizer is not tokenizer_before
            or identity_before != identity_after
        ):
            return {"supported": False, "reason": "model_changed"}
        return {
            "supported": True,
            "version": 1,
            "tokens": tokens,
            "count": len(tokens),
            "model": manager.served_model_name,
            "model_path": manager.model_path,
            "weight_version": manager.config_value("weight_version"),
        }
