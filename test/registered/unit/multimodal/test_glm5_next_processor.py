"""CPU-only GLM5Next processing/loader regressions; no weights or Hub access."""

import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast
from transformers.video_utils import VideoMetadata

from sglang.srt.multimodal.glm5_next.processing_glm5_next import Glm5NextProcessor
from sglang.srt.utils.hf_transformers.processor import get_processor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

TOKENS = [
    "<unk>",
    "<pad>",
    "<|image|>",
    "<|video|>",
    "<|begin_of_image|>",
    "<|end_of_image|>",
    "<|begin_of_video|>",
    "<|end_of_video|>",
]
TEMPLATE = """{% for m in messages %}{{ m.role }}: {% if m.content is string %}{{ m.content }}{% else %}{% for c in m.content %}{% if c.type in ['image', 'image_url'] %}<|begin_of_image|><|image|><|end_of_image|>{% elif c.type == 'text' %}{{ c.text }}{% endif %}{% endfor %}{% endif %}{% endfor %}"""


def png_data_url(size, color):
    image = Image.new("RGB", size, color)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def decode_data_url(url):
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")


class TestGlm5NextProcessor(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)
        backend = Tokenizer(
            WordLevel(dict(zip(TOKENS, range(len(TOKENS)))), unk_token="<unk>")
        )
        backend.pre_tokenizer = WhitespaceSplit()
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="<unk>",
            pad_token="<pad>",
            additional_special_tokens=TOKENS[2:],
            chat_template=TEMPLATE,
        )
        self.tokenizer.save_pretrained(self.path)
        self.config = {
            "model_type": "glm5_next",
            "architectures": ["Glm5NextForConditionalGeneration"],
            "text_config": {
                "hidden_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
            },
            "vision_config": {
                "patch_size": 14,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "swiglu_limit": 10.0,
            },
        }
        self.processor_config = {
            "processor_class": "Glm5NextProcessor",
            "image_processor": {
                "image_processor_type": "Glm5NextImageProcessor",
                "patch_size": 14,
                "temporal_patch_size": 2,
                "merge_size": 2,
                "min_image_tokens": 16,
                "max_image_tokens": 8000,
            },
            "video_processor": {
                "video_processor_type": "Glm5NextVideoProcessor",
                "patch_size": 14,
                "temporal_patch_size": 2,
                "merge_size": 2,
                "fps": 2,
            },
        }
        self.write_configs()
        self.processor = Glm5NextProcessor.from_pretrained(
            self.path, local_files_only=True
        )

    def write_configs(self):
        (self.path / "config.json").write_text(json.dumps(self.config))
        (self.path / "processor_config.json").write_text(
            json.dumps(self.processor_config)
        )

    def load(self, **kwargs):
        return get_processor(str(self.path), local_files_only=True, **kwargs)

    def test_auto_loader_returns_multimodal_processor(self):
        processor = self.load()
        self.assertIsNotNone(processor.image_processor)
        self.assertEqual(processor.image_processor.temporal_patch_size, 2)
        result = processor(
            text="<|image|>",
            images=Image.new("RGB", (137, 93), "red"),
            return_tensors="pt",
        )
        self.assertEqual(result["pixel_values"].shape, (80, 1176))
        self.assertEqual(
            int((result["input_ids"] == processor.image_token_id).sum()), 20
        )

    def test_tokenizer_only_success_triggers_fallback(self):
        with patch(
            "sglang.srt.utils.hf_transformers.processor.AutoProcessor.from_pretrained",
            return_value=self.tokenizer,
        ):
            self.assertIsInstance(self.load(), Glm5NextProcessor)

    def test_unknown_processor_error_triggers_fallback(self):
        with patch(
            "sglang.srt.utils.hf_transformers.processor.AutoProcessor.from_pretrained",
            side_effect=ValueError("Unknown processor"),
        ):
            self.assertIsInstance(self.load(), Glm5NextProcessor)

    def test_native_processor_is_preserved(self):
        with patch(
            "sglang.srt.utils.hf_transformers.processor.AutoProcessor.from_pretrained",
            return_value=self.processor,
        ):
            self.assertIs(self.load(), self.processor)

    def test_missing_image_config_fails_loudly(self):
        self.processor_config.pop("image_processor")
        self.write_configs()
        with self.assertRaisesRegex(ValueError, "requires image_processor"):
            self.load()

    def test_geometry_mismatch_fails_loudly(self):
        self.processor_config["image_processor"]["temporal_patch_size"] = 1
        self.write_configs()
        with self.assertRaisesRegex(ValueError, "disagrees with vision_config"):
            self.load()

    def test_explicit_backend(self):
        self.assertEqual(
            self.load(image_processor_backend="torchvision").image_processor.backend,
            "torchvision",
        )
        with self.assertRaisesRegex(ValueError, "requires the torchvision backend"):
            self.load(image_processor_backend="pil")

    def test_base64_two_images_multi_turn_order_and_counts(self):
        urls = [png_data_url((137, 93), "red"), png_data_url((84, 168), "blue")]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": urls[0]}},
                    {"type": "text", "text": "Describe this"},
                ],
            },
            {"role": "assistant", "content": "Red"},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": urls[1]}},
                    {"type": "text", "text": "Compare with the previous image"},
                ],
            },
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False)
        images = [decode_data_url(u) for u in urls]
        result = self.processor(text=text, images=images, return_tensors="pt")
        self.assertEqual(result["image_grid_thw"].tolist(), [[1, 8, 10], [1, 12, 6]])
        self.assertEqual(result["pixel_values"].shape, (152, 1176))
        self.assertEqual(
            int((result["input_ids"] == self.processor.image_token_id).sum()), 38
        )
        self.assertEqual(int((result["mm_token_type_ids"] == 1).sum()), 38)
        for i, part in enumerate(result["pixel_values"].split([80, 72])):
            single = self.processor.image_processor(
                images=images[i], return_tensors="pt"
            )["pixel_values"]
            torch.testing.assert_close(part, single, rtol=0, atol=0)
        # Caller text remains reusable; no in-place expansion across requests.
        repeated = self.processor(text=text, images=images, return_tensors="pt")
        torch.testing.assert_close(result["input_ids"], repeated["input_ids"])

    def test_patch_order_temporal_duplication_and_clip_padding(self):
        pixels = (
            np.arange(93 * 137 * 3, dtype=np.uint32)
            .reshape(93, 137, 3)
            .astype(np.uint8)
        )
        result = self.processor.image_processor(
            Image.fromarray(pixels), return_tensors="pt"
        )
        # Undo merge-major patch ordering independently, checking every pixel.
        patches = result["pixel_values"].reshape(4, 5, 2, 2, 3, 2, 14, 14)
        canvas = patches.permute(4, 5, 0, 2, 6, 1, 3, 7).reshape(3, 2, 112, 140)
        expected = torch.zeros(3, 112, 140)
        expected[:, :93, :137] = torch.from_numpy(pixels.copy()).permute(2, 0, 1) / 255
        mean = torch.tensor(self.processor.image_processor.image_mean)[:, None, None]
        std = torch.tensor(self.processor.image_processor.image_std)[:, None, None]
        expected = (expected - mean) / std
        torch.testing.assert_close(canvas[:, 0], expected)
        torch.testing.assert_close(canvas[:, 1], expected)

    def test_resize_budget_and_patch_estimator(self):
        for size in [(13, 19), (137, 93), (1400, 700), (8, 800)]:
            with self.subTest(size=size):
                result = self.processor.image_processor(
                    Image.new("RGB", size), max_image_tokens=32, return_tensors="pt"
                )
                patches = int(result["image_grid_thw"].prod())
                self.assertLessEqual(patches // 4, 32)
                self.assertEqual(
                    patches,
                    self.processor.image_processor.get_number_of_image_patches(
                        size[1], size[0], {"max_image_tokens": 32}
                    ),
                )

    def test_placeholder_mismatch_and_truncation_rejected(self):
        image = Image.new("RGB", (137, 93))
        for text, images in [
            ("describe", [image]),
            ("<|image|>", [image, image]),
            ("<|image|><|image|>", [image]),
            ("<|image|>", None),
        ]:
            with self.subTest(
                text=text, count=len(images or [])
            ), self.assertRaisesRegex(ValueError, "placeholders"):
                self.processor(text=text, images=images)
        with self.assertRaisesRegex(ValueError, "lost during tokenization"):
            self.processor(
                text="<|image|>", images=image, truncation=True, max_length=5
            )

    def test_video_odd_frames_and_mixed_image_do_not_consume_each_other(self):
        frames = np.stack([np.full((112, 112, 3), c, np.uint8) for c in [0, 127, 255]])
        metadata = VideoMetadata(
            total_num_frames=3, fps=2.0, duration=1.5, frames_indices=[0, 1, 2]
        )
        result = self.processor(
            text="<|begin_of_image|><|image|><|end_of_image|><|begin_of_video|><|video|><|end_of_video|><|image|>",
            images=[
                Image.new("RGB", (112, 112), "red"),
                Image.new("RGB", (112, 112), "blue"),
            ],
            videos=[frames],
            do_sample_frames=False,
            video_metadata=[metadata],
            return_tensors="pt",
        )
        self.assertEqual(result["video_grid_thw"].tolist(), [[2, 8, 8]])
        self.assertEqual(result["pixel_values_videos"].shape, (128, 1176))
        self.assertEqual(int((result["mm_token_type_ids"] == 1).sum()), 32)
        self.assertEqual(int((result["mm_token_type_ids"] == 2).sum()), 32)
        packed = result["pixel_values_videos"].reshape(2, 64, 3, 2, 14, 14)
        torch.testing.assert_close(packed[1, :, :, 0], packed[1, :, :, 1])
        self.assertEqual(
            self.processor.replace_frame_token_id(1.5),
            "<|begin_of_image|><|image|><|end_of_image|>1.5 seconds",
        )

    def test_video_sampling_and_estimator(self):
        video = self.processor.video_processor
        metadata = VideoMetadata(total_num_frames=9, fps=4.0, duration=2.25)
        self.assertEqual(video.sample_frames(metadata).tolist(), [0, 2, 4, 6])
        self.assertEqual(
            self.processor._get_num_multimodal_tokens(
                video_sizes=[(3, 112, 112)]
            ).num_video_tokens,
            [32],
        )


if __name__ == "__main__":
    unittest.main()
