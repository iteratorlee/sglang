# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import logging
import json
from pathlib import Path
from contextlib import contextmanager
from copy import copy

import torch

from sglang.srt.model_loader.weight_utils import default_weight_loader

from sglang.srt.models.deepseek_nextn import DeepseekV3ForCausalLMNextN
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.srt.models.utils import WeightsMapper

logger = logging.getLogger(__name__)


class Glm5NextForConditionalGenerationNextN(DeepseekV3ForCausalLMNextN):
    requires_aligned_npu_mtp_graph = True
    npu_draft_extend_low_latency = True
    validate_weights_after_loading = (
        Glm5NextForConditionalGeneration.validate_weights_after_loading
    )

    @classmethod
    def get_hf_to_sglang_mapper(cls, config) -> WeightsMapper:
        text_config = getattr(config, "text_config", config)
        return WeightsMapper(
            orig_to_new_prefix={
                f"model.language_model.layers.{text_config.num_hidden_layers}": "model.decoder",
                f"model.layers.{text_config.num_hidden_layers}": "model.decoder",
            },
            orig_to_new_substr=Glm5NextForConditionalGeneration.hf_to_sglang_mapper.orig_to_new_substr,
        )

    def _resolve_nextn_quant_config(self, config, quant_config):
        """Mixed checkpoints list the BF16 NextN block in ``quantization_config.ignore``;
        inheriting global FP8 quantization would corrupt its QKV weights."""
        raw_quant_config = getattr(config, "quantization_config", None) or {}
        if hasattr(raw_quant_config, "to_dict"):
            raw_quant_config = raw_quant_config.to_dict()
        ignored = (
            raw_quant_config.get("ignore", [])
            if isinstance(raw_quant_config, dict)
            else []
        )
        nextn_layer_pattern = f"model.layers.{config.num_hidden_layers}.*"
        if nextn_layer_pattern in ignored:
            logger.warning(
                "GLM5 NextN layer %s is checkpoint-declared unquantized; "
                "using BF16 draft modules",
                nextn_layer_pattern,
            )
            return None
        return super()._resolve_nextn_quant_config(config, quant_config)

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        config = copy(getattr(config, "text_config", config))
        if config.qk_rope_head_dim == 0:
            # Reuse native NextN's NoPE decoder path, including its indexer.
            config.mla_nope = True
        self._modelslim_checkpoint = (
            quant_config is not None and quant_config.get_name() == "modelslim"
        )
        super().__init__(
            config,
            quant_config=quant_config,
            prefix=prefix,
        )
        if self._modelslim_checkpoint:
            if not quant_config.quant_description.get("is_rot_used"):
                raise ValueError("GLM ModelSlim MTP requires its checkpoint rotation")
            self.model.zero_position_embeddings = True
            self.model.decoder.self_attn.indexer._glm53_draft_decode = True
            from sglang.srt.runtime_context import get_spec

            self.model.decoder.self_attn.indexer._glm53_draft_graph_steps = (
                get_spec().speculative_num_draft_tokens
            )

    @contextmanager
    def speculative_state_transaction(self):
        indexer = self.model.decoder.self_attn.indexer
        if not self._modelslim_checkpoint or not hasattr(indexer, "_kpool_tail_k"):
            yield
            return
        keys = indexer._kpool_tail_k.clone()
        scores = indexer._kpool_tail_score.clone()
        try:
            yield
        finally:
            # Draft guesses must not advance the persistent partial KPool.
            # The accepted-token extension writes its chosen prefix later.
            indexer._kpool_tail_k.copy_(keys)
            indexer._kpool_tail_score.copy_(scores)

    def get_local_weight_iterator(self, model_path):
        """Read only the checkpoint MTP tensors and shared rotation."""
        if not self._modelslim_checkpoint:
            return None
        from safetensors import safe_open

        folder = Path(model_path)
        path = folder / "quant_model_weights.safetensors.index.json"
        if not path.is_file():
            return None
        index = json.loads(path.read_text())["weight_map"]
        layer = self.config.num_hidden_layers
        prefixes = (f"model.layers.{layer}.", f"model.language_model.layers.{layer}.")
        selected = {}
        for name, shard in index.items():
            if name.startswith(prefixes) or name == "rot.weight":
                selected.setdefault(shard, []).append(name)
        if not selected:
            raise ValueError("GLM checkpoint contains no MTP tensors")
        logger.info(
            "GLM MTP selects %d tensors from %d shards",
            sum(map(len, selected.values())),
            len(selected),
        )

        def iterator():
            for shard, names in sorted(selected.items()):
                with safe_open(
                    str(folder / shard), framework="pt", device="cpu"
                ) as handle:
                    for name in names:
                        yield name, handle.get_tensor(name)

        return iterator()

    def set_embed_and_head(self, embed, head):
        if not self._modelslim_checkpoint:
            super().set_embed_and_head(embed, head)
        # Rotated ModelSlim exports contain a distinct MTP embedding/head.

    def load_weights(self, weights):
        if not hasattr(self, "fuse_qkv_a_proj"):
            self.fuse_qkv_a_proj = getattr(self.config, "q_lora_rank", None) is not None
        layer_id = self.config.num_hidden_layers
        layer_prefixes = (
            f"model.layers.{layer_id}.",
            f"model.language_model.layers.{layer_id}.",
        )
        loaded = set()

        def nextn_weights():
            for name, weight in weights:
                canonical = name.replace("model.language_model.", "model.")
                if self._modelslim_checkpoint and canonical == "rot.weight":
                    # Native NextN uses matmul; checkpoint Linear uses W.T.
                    self.model.rot_weight = weight.T.contiguous().to(
                        self.model.embed_tokens.weight
                    )
                    loaded.add("rotation")
                elif (
                    self._modelslim_checkpoint
                    and canonical == f"model.layers.{layer_id}.embed_tokens.weight"
                ):
                    param = self.model.embed_tokens.weight
                    getattr(param, "weight_loader", default_weight_loader)(
                        param, weight
                    )
                    loaded.add("embedding")
                elif (
                    self._modelslim_checkpoint
                    and canonical == f"model.layers.{layer_id}.shared_head.head.weight"
                ):
                    param = self.lm_head.weight
                    getattr(param, "weight_loader", default_weight_loader)(
                        param, weight
                    )
                    loaded.add("head")
                elif name.startswith(layer_prefixes):
                    yield name, weight

        result = Glm5NextForConditionalGeneration.load_weights(
            self, nextn_weights(), is_nextn=True
        )
        if self._modelslim_checkpoint and loaded != {"rotation", "embedding", "head"}:
            raise ValueError(f"Incomplete GLM MTP checkpoint tensors: {loaded}")
        return result


EntryClass = [Glm5NextForConditionalGenerationNextN]
