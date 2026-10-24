# Copyright 2026 the HuggingFace Team. All rights reserved.
# Copyright 2026 SGLang Team
#
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
"""GLM5Next processor backport for Transformers without native GLM5Next.

Adapted from Hugging Face's Apache-2.0 processing_glm5_next.py:
https://github.com/huggingface/transformers/tree/main/src/transformers/models/glm5_next
Uses GLM4v's compatible initialization and modality IDs, with explicit expansion
for older ProcessorMixin versions and strict media/token count validation.
"""

import json
import re

from transformers import AutoTokenizer, Glm4vProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.glm4v.processing_glm4v import Glm4vProcessorKwargs
from transformers.processing_utils import MultiModalData
from transformers.utils.hub import cached_file

from .image_processing_glm5_next import Glm5NextImageProcessor
from .video_processing_glm5_next import Glm5NextVideoProcessor


class Glm5NextProcessor(Glm4vProcessor):
    @classmethod
    def from_pretrained(
        cls,
        model_path,
        *args,
        tokenizer_name=None,
        image_processor_backend="auto",
        **kwargs,
    ):
        if image_processor_backend not in {"auto", "torchvision"}:
            raise ValueError(
                "GLM5Next compatibility processor requires the torchvision backend"
            )
        hub_kwargs = {
            key: kwargs[key]
            for key in (
                "revision",
                "cache_dir",
                "force_download",
                "local_files_only",
                "token",
                "subfolder",
            )
            if key in kwargs
        }
        path = cached_file(model_path, "processor_config.json", **hub_kwargs)
        with open(path) as f:
            config = json.load(f)
        image_config = config.get("image_processor")
        if not isinstance(image_config, dict) or not image_config:
            raise ValueError(
                "GLM5Next requires image_processor in processor_config.json"
            )
        components = {}
        for name, component_cls in (
            ("image_processor", Glm5NextImageProcessor),
            ("video_processor", Glm5NextVideoProcessor),
        ):
            component_config = dict(config.get(name) or {})
            declared = component_config.pop(name + "_type", component_cls.__name__)
            if declared != component_cls.__name__:
                raise ValueError(f"Unsupported GLM5Next {name}: {declared}")
            component_config.pop("auto_map", None)
            component_config.pop("processor_class", None)
            component_config.update(
                {
                    k: v
                    for k, v in kwargs.items()
                    if k in component_cls.valid_kwargs.__annotations__
                }
            )
            components[name] = component_cls(**component_config)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or model_path, *args, **kwargs
        )
        return cls(
            **components, tokenizer=tokenizer, chat_template=tokenizer.chat_template
        )

    def replace_image_token(self, image_inputs, image_idx, **kwargs):
        count = (
            int(image_inputs["image_grid_thw"][image_idx].prod())
            // self.image_processor.merge_size**2
        )
        return self.image_token * count

    def replace_video_token(self, video_inputs, video_idx, **kwargs):
        grid = video_inputs["video_grid_thw"][video_idx]
        frames = int(grid[0])
        count = int(grid.prod()) // self.video_processor.merge_size**2 // frames
        metadata = video_inputs["video_metadata"][video_idx]
        if metadata.fps is None or metadata.fps <= 0:
            raise ValueError(
                "GLM5Next video timestamps require positive fps in video_metadata"
            )
        timestamps = list(
            metadata.timestamps[:: self.video_processor.temporal_patch_size]
        )
        timestamps += [timestamps[-1] if timestamps else 0] * (frames - len(timestamps))
        return "".join(
            self.replace_frame_token_id(t, count) for t in timestamps[:frames]
        )

    def replace_frame_token_id(self, timestamp_sec, num_image_tokens=1):
        return f"<|begin_of_image|>{self.image_token * num_image_tokens}<|end_of_image|>{timestamp_sec:.1f} seconds"

    def __call__(self, images=None, text=None, videos=None, **kwargs):
        output_kwargs = self._merge_kwargs(
            Glm4vProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        image_inputs = (
            self.image_processor(images=images, **output_kwargs["images_kwargs"])
            if images is not None
            else {}
        )
        video_inputs = (
            self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            if videos is not None
            else {}
        )
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_ids = output_kwargs["text_kwargs"].pop(
            "return_mm_token_type_ids", False
        )
        text_inputs = {}
        if text is not None:
            texts = [text] if isinstance(text, str) else list(text)
            replacements = {}
            for token, inputs, grid_key, replace in (
                (
                    self.image_token,
                    image_inputs,
                    "image_grid_thw",
                    self.replace_image_token,
                ),
                (
                    self.video_token,
                    video_inputs,
                    "video_grid_thw",
                    self.replace_video_token,
                ),
            ):
                count = len(inputs.get(grid_key, []))
                placeholders = sum(t.count(token) for t in texts)
                if placeholders != count:
                    raise ValueError(
                        f"GLM5Next {token}: {placeholders} placeholders for {count} media items"
                    )
                replacements[token] = iter([replace(inputs, i) for i in range(count)])
            # Expand both modalities in one pass: video frame image tokens must
            # never consume an image belonging to an earlier/later chat turn.
            pattern = re.compile(
                f"{re.escape(self.image_token)}|{re.escape(self.video_token)}"
            )
            texts = [pattern.sub(lambda m: next(replacements[m[0]]), t) for t in texts]
            text_inputs = self.tokenizer(texts, **output_kwargs["text_kwargs"])
            for expanded, ids in zip(texts, text_inputs["input_ids"]):
                if expanded.count(self.image_token) != list(ids).count(
                    self.image_token_id
                ):
                    raise ValueError(
                        "GLM5Next image tokens were lost during tokenization or truncation"
                    )
            if return_mm_ids:
                text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(
                    text_inputs["input_ids"]
                )
        elif images is None and videos is None:
            raise ValueError("GLM5Next requires text, images or videos")
        if not kwargs.get("return_metadata"):
            video_inputs.pop("video_metadata", None)
        return BatchFeature(
            data={**text_inputs, **image_inputs, **video_inputs},
            tensor_type=return_tensors,
        )

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        data = {}
        if image_sizes is not None:
            patches = [
                self.image_processor.get_number_of_image_patches(*size, kwargs)
                for size in image_sizes
            ]
            data["num_image_patches"] = patches
            data["num_image_tokens"] = [
                n // self.image_processor.merge_size**2 for n in patches
            ]
        if video_sizes is not None:
            patches = [
                self.video_processor.get_num_of_video_patches(*size, kwargs)
                for size in video_sizes
            ]
            data["num_video_tokens"] = [
                n // self.video_processor.merge_size**2 for n in patches
            ]
        return MultiModalData(**data)
